def gradfilter_with_depth_scaling(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    alpha: float = 0.98,
    lamb_max: float = 3.0,
    lamb_min: float = 1.0,
    d_max: int = 12,  # Total number of transformer layers
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,
    embedding_layer_name: str = "embedding",
    final_and_output_layer_names: List[str] = [
        "ln_f",
        "head",
    ],  # Default final and output layer names
) -> Dict[str, deque]:
    """
    Applies gradient filtering with dynamic depth-based lambda scaling.

    Args:
        m (nn.Module): The model containing the parameters.
        grads (Optional[Dict[str, deque]]): Dictionary for storing past gradients.
        window_size (int): Number of past gradients to consider.
        lamb_max (float): Maximum lambda value for scaling.
        lamb_min (float): Minimum lambda value for scaling.
        d_max (int): Total depth (number of transformer layers).
        filter_type (Literal['mean', 'sum']): Filtering strategy ('mean' or 'sum').
        warmup (bool): Whether to enable warmup for gradient filtering.
        trigger (bool): Optional trigger condition for gradient filtering.
        embedding_layer_name (str): Substring identifying embedding layer parameters.
        final_and_output_layer_names (List[str]): List of substrings identifying final/output layer parameters.

    Returns:
        Dict[str, deque]: Updated gradient storage for the model parameters.
    """
    if grads is None:
        grads = {
            n: p.grad.data.clone()
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            # Determine depth or position
            if embedding_layer_name in n:
                depth = 0  # Embedding layers are assigned depth 0
            elif "layers" in n:
                # Extract depth information from name, e.g., "layers.0", "layers.1"
                depth = (
                    int(n.split(".")[1]) + 1
                )  # Increment depth for transformer layers
            elif final_and_output_layer_names and any(
                layer_name in n for layer_name in final_and_output_layer_names
            ):
                depth = d_max + 1  # Final and output layers are d_max + 1
            else:
                depth = d_max  # Default depth for unclassified layers

            # Adjust lambda based on depth
            lambda_d = lamb_max - (depth / (d_max + 1)) * (lamb_max - lamb_min)

            # Apply EMA update
            if n not in grads:
                grads[n] = p.grad.data.clone()  # Initialize EMA
            else:
                grads[n] = grads[n] * alpha + p.grad.data.clone() * (1 - alpha)

            # Scale gradient by depth-aware lambda
            p.grad.data = p.grad.data + grads[n] * lambda_d

    return grads
