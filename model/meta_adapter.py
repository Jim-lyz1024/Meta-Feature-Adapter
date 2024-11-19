import torch
import torch.nn as nn 
from torch.nn import functional as F

class MetaTextGenerator(nn.Module):
    """
    Generates text templates for different metadata types 
    and converts them to text features using CLIP's text encoder
    """
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        # Temperature ranges for categorization 
        self.temp_ranges = {
            'cold': (-float('inf'), 15),
            'mild': (15, 25), 
            'hot': (25, float('inf'))
        }

        # Humidity ranges
        self.humid_ranges = {
            'dry': (-float('inf'), 60),
            'moderate': (60, 80),
            'humid': (80, float('inf'))
        }

        # Rain ranges
        self.rain_ranges = {
            'no_rain': 0,
            'light': (0, 10),
            'moderate': (10, 50),
            'heavy': (50, float('inf'))
        }

        # Angle mappings
        self.angle_map = {
            0: 'front',
            1: 'back', 
            2: 'left',
            3: 'right'
        }

    def _get_temp_category(self, temp):
        for cat, (low, high) in self.temp_ranges.items():
            if low <= temp < high:
                return cat
        return 'mild'  # Default case

    def _get_humid_category(self, humid):
        for cat, (low, high) in self.humid_ranges.items():
            if low <= humid < high:
                return cat
        return 'moderate'  # Default case

    def _get_rain_category(self, rain):
        if rain == 0:
            return 'no_rain'
        for cat, (low, high) in self.rain_ranges.items():
            if low <= rain < high:
                return cat
        return 'light'  # Default case

    def _get_angle_category(self, angle):
        return self.angle_map.get(angle, 'front')  # Default to 'front'

    def encode_text(self, text_tokens):
        """Convert text tokens to features using CLIP's text encoder components"""
        x = self.token_embedding(text_tokens).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection
        return x

    def generate_text_features(self, metadata):
        """Generate text features for all metadata types in a batch."""
        from .clip.simple_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()

        batch_size = metadata['temperature'].shape[0]
        device = metadata['temperature'].device

        # Initialize lists to store templates and tokens
        templates = []

        for i in range(batch_size):
            # Extract scalar values from tensors
            temp = metadata['temperature'][i].item()
            humid = metadata['humidity'][i].item()
            rain = metadata['rain'][i].item()
            angle = metadata['angle'][i].item()

            # Get categories
            temp_cat = self._get_temp_category(temp)
            humid_cat = self._get_humid_category(humid)
            rain_cat = self._get_rain_category(rain)
            angle_cat = self._get_angle_category(angle)

            # Generate a combined template for the sample
            template = (
                f"A photo of a stoat in {temp_cat} temperature, "
                f"{humid_cat} humidity, "
                f"{rain_cat} precipitation, "
                f"from the {angle_cat}."
            )
            templates.append(template)

        # Tokenize and encode the templates
        text_tokens = []
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]

        for template in templates:
            encoded = tokenizer.encode(template)
            token_sequence = [sot_token] + encoded + [eot_token]
            text_tokens.append(torch.tensor(token_sequence, dtype=torch.long))

        # Pad sequences to the same length
        max_len = max(len(t) for t in text_tokens)
        padded_tokens = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, tokens in enumerate(text_tokens):
            padded_tokens[i, :len(tokens)] = tokens

        padded_tokens = padded_tokens.to(device)

        # Encode text features
        text_features = self.encode_text(padded_tokens)
        return text_features

class CrossAttention(nn.Module):
    """
    Cross attention module to combine image and text features
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MetaAdapter(nn.Module):
    """
    Main MetaAdapter module that combines image features with metadata
    through text templates and cross attention
    """
    def __init__(self, clip_model, embed_dim=768):
        super().__init__()
        self.text_generator = MetaTextGenerator(clip_model)
        self.cross_attn = CrossAttention(embed_dim)

    def forward(self, image_features, metadata):
        """
        Args:
            image_features: Features from CLIP's image encoder
            metadata: Dict containing temperature, humidity, rain and angle values
        """
        # Generate text features from metadata
        text_features = self.text_generator.generate_text_features(metadata)
        
        # Reshape image features if necessary 
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)
            
        # Reshape text features if necessary
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
            
        # Combine features using cross attention
        enhanced_features = self.cross_attn(image_features, text_features)
        
        return enhanced_features.squeeze(1) # Remove sequence dimension for final output