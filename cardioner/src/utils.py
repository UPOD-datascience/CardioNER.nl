import pprint
import torch.nn as nn

def pretty_print_classifier(classifier):
    """Pretty print a PyTorch module with indentation and structure details."""
    import pprint
    import torch.nn as nn

    pp = pprint.PrettyPrinter(indent=4, width=100)

    if isinstance(classifier, nn.Linear):
        return f"Linear(in_features={classifier.in_features}, out_features={classifier.out_features})"

    if isinstance(classifier, nn.Sequential):
        # Create a structured representation as a dictionary
        layers_info = {}
        for i, layer in enumerate(classifier):
            layer_info = {
                'type': layer.__class__.__name__
            }

            # Add details based on layer type
            if isinstance(layer, nn.Linear):
                layer_info['in_features'] = layer.in_features
                layer_info['out_features'] = layer.out_features
            elif isinstance(layer, nn.Dropout):
                layer_info['p'] = layer.p
            elif isinstance(layer, nn.BatchNorm1d):
                layer_info['num_features'] = layer.num_features
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                layer_info['in_channels'] = layer.in_channels
                layer_info['out_channels'] = layer.out_channels

            layers_info[f'layer_{i}'] = layer_info

        # Use pprint to format the dictionary
        return f"Sequential with {len(classifier)} layers:\n{pp.pformat(layers_info)}"

    # For other module types, convert to dict and pretty print
    try:
        # Try to extract meaningful attributes
        module_info = {
            'type': classifier.__class__.__name__,
            'parameters': {name: param.size() for name, param in classifier.named_parameters()}
        }
        return pp.pformat(module_info)
    except:
        # Fallback
        return str(classifier)
