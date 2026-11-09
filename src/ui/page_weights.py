
"""Display FP32 and INT8 CNN weights with filter/channel kernel visualization."""

import pandas as pd
import streamlit as st
import torch

from src.ui.common import get_predictor


def quantize_weights(tensor):
    """Convert FP32 weights into signed INT8 representation."""

    max_abs = tensor.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    zero_point = 0

    quantized = torch.quantize_per_tensor(
        tensor,
        scale=scale,
        zero_point=zero_point,
        dtype=torch.qint8,
    )

    int8_values = quantized.int_repr()
    dequantized = quantized.dequantize()

    return int8_values, dequantized, scale, zero_point


def show_kernel(fp32_kernel, int8_kernel):
    """Display the selected kernel as FP32 and INT8 matrices."""

    fp32_matrix = fp32_kernel.detach().cpu().numpy()
    int8_matrix = int8_kernel.detach().cpu().numpy()

    rows = [f"Row {i + 1}" for i in range(fp32_matrix.shape[0])]
    cols = [f"Col {i + 1}" for i in range(fp32_matrix.shape[1])]

    fp32_df = pd.DataFrame(
        fp32_matrix,
        index=rows,
        columns=cols,
    )

    int8_df = pd.DataFrame(
        int8_matrix,
        index=rows,
        columns=cols,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### FP32 Kernel")
        st.caption("Original floating-point values")
        st.dataframe(
            fp32_df.style.format("{:.6f}"),
            use_container_width=True,
        )

    with col2:
        st.markdown("### INT8 Kernel")
        st.caption("Quantized integer values")
        st.dataframe(
            int8_df,
            use_container_width=True,
        )


def render(cfg: dict) -> None:

    st.title("CNN Weights")

    st.write(
        "Explore actual learned CNN weights in FP32 and "
        "their INT8 quantized representations."
    )

    predictor, error = get_predictor(cfg)

    if predictor is None:
        st.error(f"Unable to load the trained CNN: {error}")
        return

    model = predictor.model
    state_dict = model.state_dict()

    weight_names = [
        name for name, tensor in state_dict.items()
        if name.endswith(".weight") and tensor.is_floating_point()
    ]

    if not weight_names:
        st.warning("No learned weight tensors were found.")
        return

    st.caption(f"Model version: {predictor.model_version}")

    total_weights = sum(
        state_dict[name].numel() for name in weight_names
    )

    fp32_size = total_weights * 4
    int8_size = total_weights

    col1, col2, col3 = st.columns(3)

    col1.metric("Weight tensors", len(weight_names))
    col2.metric("Learned weights", f"{total_weights:,}")
    col3.metric("FP32 storage", f"{fp32_size / (1024 * 1024):.2f} MB")

    st.subheader("FP32 vs INT8 Storage")

    c1, c2, c3 = st.columns(3)

    c1.metric("FP32", f"{fp32_size / (1024 * 1024):.2f} MB")
    c2.metric("INT8", f"{int8_size / (1024 * 1024):.2f} MB")
    c3.metric("Estimated reduction", "75%")

    st.caption(
        "Estimated storage for learned weight values only. "
        "Quantization metadata and other model parameters are excluded."
    )

    st.divider()

    # Select CNN layer
    selected_name = st.selectbox(
        "Select a CNN layer",
        weight_names,
    )

    tensor = state_dict[selected_name].detach().cpu().float()

    st.subheader(selected_name)

    st.write(f"**Tensor shape:** {tuple(tensor.shape)}")
    st.write(f"**Number of weights:** {tensor.numel():,}")

    # Quantize the complete selected layer
    int8_tensor, dequantized_tensor, scale, zero_point = (
        quantize_weights(tensor)
    )

    st.subheader("Quantization Information")

    q1, q2, q3 = st.columns(3)

    q1.metric("Precision", "INT8")
    q2.metric("Scale", f"{scale:.8f}")
    q3.metric("Zero point", zero_point)

    # Filter and channel selection for convolution weights
    if tensor.ndim == 4:

        output_filters, input_channels, kernel_h, kernel_w = tensor.shape

        st.divider()
        st.subheader("Filter and Channel Selection")

        col1, col2 = st.columns(2)

        with col1:
            filter_index = st.selectbox(
                "Select Filter",
                range(output_filters),
                format_func=lambda x: f"Filter {x + 1}",
            )

        with col2:
            channel_index = st.selectbox(
                "Select Channel",
                range(input_channels),
                format_func=lambda x: f"Channel {x + 1}",
            )

        st.info(
            f"Showing Filter {filter_index + 1}, "
            f"Channel {channel_index + 1}"
        )

        # Extract the exact kernel
        fp32_kernel = tensor[
            filter_index, channel_index
        ]

        int8_kernel = int8_tensor[
            filter_index, channel_index
        ]

        st.subheader(
            f"{kernel_h} × {kernel_w} Kernel Values"
        )

        show_kernel(fp32_kernel, int8_kernel)

        if kernel_h == 3 and kernel_w == 3:
            st.success(
                "Displaying the selected 3×3 convolution kernel."
            )

        elif kernel_h == 1 and kernel_w == 1:
            st.caption("This layer uses a 1×1 convolution kernel.")

        else:
            st.caption(
                f"This layer uses a {kernel_h}×{kernel_w} kernel."
            )

    else:
        st.info(
            "Filter and channel selection is available for "
            "4D convolution weight tensors."
        )

    st.divider()

    # Full weight comparison
    st.subheader("Complete Weight Comparison")

    fp32_values = tensor.numpy().reshape(-1)
    int8_values = int8_tensor.numpy().reshape(-1)
    dequantized_values = dequantized_tensor.numpy().reshape(-1)

    max_error = abs(
        fp32_values - dequantized_values
    ).max()

    st.metric(
        "Maximum quantization error",
        f"{max_error:.8f}",
    )

    preview_limit = 1000
    preview_count = min(len(fp32_values), preview_limit)

    st.caption(
        f"Previewing {preview_count:,} of "
        f"{len(fp32_values):,} weight values."
    )

    preview_df = pd.DataFrame({
        "Weight index": range(preview_count),
        "FP32 weight": fp32_values[:preview_count],
        "INT8 weight": int8_values[:preview_count],
        "Dequantized weight": dequantized_values[:preview_count],
    })

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download all weights
    full_df = pd.DataFrame({
        "Weight index": range(len(fp32_values)),
        "FP32 weight": fp32_values,
        "INT8 weight": int8_values,
        "Dequantized weight": dequantized_values,
    })

    safe_name = selected_name.replace(".", "_")

    st.download_button(
        label="Download FP32 and INT8 weights CSV",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_name}_FP32_INT8_weights.csv",
        mime="text/csv",
    )
