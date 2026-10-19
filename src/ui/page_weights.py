
"""Display the actual learned weights of the deployed SteelCNN model."""

import pandas as pd
import streamlit as st

from src.ui.common import get_predictor


def render(cfg: dict) -> None:
    st.title("CNN Weights")
    st.write(
        "Explore the actual learned weight parameters from the "
        "trained CNN checkpoint used by this application."
    )

    predictor, error = get_predictor(cfg)

    if predictor is None:
        st.error(f"Unable to load the trained CNN: {error}")
        return

    model = predictor.model
    state_dict = model.state_dict()

    # Include learned weight tensors, excluding biases and other parameters.
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

    col1, col2 = st.columns(2)
    col1.metric("Weight tensors", len(weight_names))
    col2.metric("Learned weight values", f"{total_weights:,}")

    selected_name = st.selectbox(
        "Select a CNN layer",
        weight_names,
    )

    # Read values from the loaded, trained model.
    tensor = state_dict[selected_name].detach().cpu()
    values = tensor.numpy().reshape(-1)

    st.subheader(selected_name)
    st.write(f"**Tensor shape:** {tuple(tensor.shape)}")
    st.write(f"**Number of weights:** {tensor.numel():,}")

    # Show a manageable preview of the actual values.
    preview_limit = 1000
    preview_count = min(len(values), preview_limit)

    st.caption(
        f"Previewing {preview_count:,} of {len(values):,} values. "
        "Download the CSV to inspect all values."
    )

    preview_df = pd.DataFrame({
        "Weight index": range(preview_count),
        "Learned weight value": values[:preview_count],
    })

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
    )

    # Export every weight in the selected tensor.
    full_df = pd.DataFrame({
        "Weight index": range(len(values)),
        "Learned weight value": values,
    })

    safe_name = selected_name.replace(".", "_")

    st.download_button(
        label="Download all weights as CSV",
        data=full_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_name}_weights.csv",
        mime="text/csv",
    )
