
import json
from collections import Counter
from pathlib import Path

import streamlit as st
import torch
import torchvision
import pandas as pd
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fasterrcnn_grouped_best.pth"
CATEGORIES_PATH = BASE_DIR / "categories.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Kirana Shop Visual Assessment",
    page_icon="🛒",
    layout="wide"
)

CUSTOM_MODEL_PATH = (
    BASE_DIR / "custom_detector" / "kirana_custom_fasterrcnn.pth"
)

CUSTOM_CATEGORIES = {
    1: "hanging_sachet_strip",
    2: "loose_grains",
}


# =========================================================
# APPROXIMATE UNIT PRICES IN INR
# These are illustrative estimates, NOT verified current MRPs.
# Edit them in the dashboard according to your package sizes.
# =========================================================

DEFAULT_PRICES = {
    "boost_std": 160,
    "clear_ahf": 100,
    "clear_cac": 100,
    "clear_csm_large": 130,
    "clear_csm_small": 75,
    "closeup_lemon_salt": 100,
    "dove_cond": 180,
    "dove_hfr_large": 250,
    "dove_hfr_small": 100,
    "dove_irp_large": 250,
    "dove_irp_small": 100,
    "dove_mask_25": 40,
    "dove_no": 70,
    "dove_nr_lotion": 180,
    "dove_oxg": 180,
    "gl_aryuvedic_crm": 100,
    "gl_foundation_crm": 120,
    "gl_insta_glow_fw": 100,
    "gl_mltvit_crm": 120,
    "gl_sunscrn_crm": 180,
    "horlicks_choco": 150,
    "horlicks_junior": 170,
    "horlicks_junior_s1": 100,
    "horlicks_lite": 190,
    "horlicks_mother": 270,
    "horlicks_std": 160,
    "horlicks_women": 280,
    "lifebuoy_pouch": 50,
    "lifebuoy_pump": 100,
    "lux_blk_orchd": 35,
    "lux_brightening_vitamin": 35,
    "lux_freeasia_scnt": 35,
    "lux_french_rose": 35,
    "maltova_std": 150,
    "pepsodent_advanced_salt": 60,
    "pepsodent_germicheck": 60,
    "pepsodent_sensitive_expert": 100,
    "ponds_bright_beauty_crm": 120,
    "ponds_bright_beauty_fw": 120,
    "ponds_oil_control_fw": 120,
    "ponds_pure_white_fw": 100,
    "sunsilk_black_large": 220,
    "sunsilk_black_small": 100,
    "sunsilk_hfs": 100,
    "sunsilk_hfs_rinse": 100,
    "sunsilk_hrr": 100,
    "sunsilk_onion": 120,
    "sunsilk_serum_25": 40,
    "sunsilk_tl_large": 220,
    "sunsilk_tl_small": 100,
    "sunsilk_volume": 120,
    "surf_excel_matic_1l": 170,
    "surf_excel_matic_500ml": 90,
    "treseme_sampoo_bond_plex": 250,
    "tresemme_ks_large": 250,
    "tresemme_ks_small": 120,
    "tresemme_ks_white": 120,
    "tresemme_mask_25": 50,
    "tresemme_serum_25": 50,
    "vaseline_aloe": 100,
    "vaseline_gluta_flawless": 150,
    "vaseline_gluta_rad": 150,
    "vaseline_hw": 100,
    "vaseline_tm": 100,
    "vim_liquid_500ml": 130
}


# =========================================================
# LOAD CATEGORY METADATA
# =========================================================

@st.cache_data
def load_categories():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        item["model_label"]: item["name"]
        for item in metadata["categories"]
    }


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        box_detections_per_img=300
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource
def load_custom_model():
    try:
        checkpoint = torch.load(
            CUSTOM_MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CUSTOM_MODEL_PATH,
            map_location=DEVICE,
        )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Background + hanging_sachet_strip + loose_grains
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        3,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

# =========================================================
# PRODUCT DETECTION
# =========================================================

def detect_products(image, model, categories, threshold):
    image_tensor = to_tensor(image).to(DEVICE)

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    detections = []

    boxes = prediction["boxes"].cpu().tolist()
    labels = prediction["labels"].cpu().tolist()
    scores = prediction["scores"].cpu().tolist()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        if label not in categories:
            continue

        detections.append({
            "box": box,
            "label": label,
            "name": categories[label],
            "score": score,
            "source": "Original detector"
        })

    return detections

def detect_custom_objects(image, model, threshold):
    image_tensor = to_tensor(image).to(DEVICE)

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    detections = []

    for box, label, score in zip(
        prediction["boxes"].cpu().tolist(),
        prediction["labels"].cpu().tolist(),
        prediction["scores"].cpu().tolist(),
    ):
        if score < threshold:
            continue

        if label not in CUSTOM_CATEGORIES:
            continue

        detections.append({
            "box": box,
            "label": label,
            "name": CUSTOM_CATEGORIES[label],
            "score": score,
            "source": "Custom detector",
        })

    return detections


# =========================================================
# DRAW DETECTION BOXES
# =========================================================

def draw_detections(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for item in detections:
        x1, y1, x2, y2 = item["box"]

        text = f'{item["name"]} {item["score"]:.2f}'
        text_box = draw.textbbox((x1, y1), text)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=3
        )

        draw.rectangle(text_box, fill="black")
        draw.text((x1, y1), text, fill="white")

    return result


# =========================================================
# DASHBOARD
# =========================================================

st.title("🛒 Kirana Shop Visual Assessment")
st.write(
    "Detect visible products, estimate quantities, and calculate "
    "the approximate retail value represented in your shop photos."
)

st.caption(f"Device: {DEVICE.type.upper()}")

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

if not CATEGORIES_PATH.exists():
    st.error(f"Category file not found: {CATEGORIES_PATH}")
    st.stop()

if not CUSTOM_MODEL_PATH.exists():
    st.error(
        f"Custom model checkpoint not found: {CUSTOM_MODEL_PATH}"
    )
    st.stop()

try:
    categories = load_categories()
    model = load_model()
    custom_model = load_custom_model()
except Exception as error:
    st.error(f"Could not load the models: {error}")
    st.stop()


# =========================================================
# EDITABLE PRICE CATALOGUE
# =========================================================

with st.expander("⚙️ Edit approximate product prices", expanded=False):
    st.write(
        "Adjust the estimated price per detected product instance. "
        "These values are starting estimates, not verified prices. "
        "Edits apply to the current app session."
    )

    price_names = list(dict.fromkeys(
        list(categories.values()) + list(CUSTOM_CATEGORIES.values())
    ))

    price_df = pd.DataFrame([
        {
            "Product category": name,
            "Estimated unit price (₹)": float(
                DEFAULT_PRICES.get(
                    name,
                    20 if name == "hanging_sachet_strip" else 0
                )
            ),
        }
        for name in price_names
    ])

    edited_prices = st.data_editor(
        price_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Product category"],
        column_config={
            "Estimated unit price (₹)": st.column_config.NumberColumn(
                "Estimated unit price (₹)",
                min_value=0.0,
                max_value=100000.0,
                step=5.0,
                format="₹%.2f"
            )
        },
        key="price_editor"
    )

    prices = dict(zip(
        edited_prices["Product category"],
        edited_prices["Estimated unit price (₹)"]
    ))

# =========================================================
# UPLOAD IMAGES AND SET CONFIDENCE
# =========================================================

uploaded_files = st.file_uploader(
    "Upload one or more shop images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

threshold = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)


# =========================================================
# ANALYZE
# =========================================================

if uploaded_files and st.button("Analyze Images", type="primary"):

    total_detections = 0
    combined_counts = Counter()
    processed_images = 0

    st.subheader("📸 Image-wise Results")

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            image = Image.open(uploaded_file).convert("RGB")

            original_detections = detect_products(
                image,
                model,
                categories,
                threshold,
            )

            custom_detections = detect_custom_objects(
              image,
             custom_model,
             threshold,
            )

            detections = original_detections + custom_detections

            annotated_image = draw_detections(image, detections)

            image_counts = Counter(
                item["name"] for item in detections
            )

            image_value = sum(
                quantity * float(prices.get(name, 0) or 0)
                for name, quantity in image_counts.items()
            )

            total_detections += len(detections)
            combined_counts.update(image_counts)
            processed_images += 1

            st.markdown(f"### {index}. {uploaded_file.name}")

            col1, col2 = st.columns([3, 2])

            with col1:
                st.image(
                    annotated_image,
                    caption="Detected products",
                    use_container_width=True
                )

            with col2:
                st.metric(
                    "Detected product instances",
                    len(detections)
                )

                st.metric(
                    "Estimated value in this photo",
                    f"₹{image_value:,.2f}"
                )

                if image_counts:
                    st.write("**Detected product breakdown**")

                    for name, quantity in image_counts.most_common():
                        unit_price = float(prices.get(name, 0) or 0)
                        subtotal = quantity * unit_price

                        st.write(
                            f"- {name}: {quantity} × "
                            f"₹{unit_price:,.2f} = ₹{subtotal:,.2f}"
                        )
                else:
                    st.info("No products detected above the threshold.")

            st.divider()

        except Exception as error:
            st.error(
                f"Could not process {uploaded_file.name}: {error}"
            )


    columns = [
        "Product",
        "Detected quantity",
        "Estimated unit price (₹)",
        "Estimated subtotal (₹)",
    ]
    rows = []

    for name, quantity in combined_counts.most_common():
        if quantity <= 0:
            continue

        unit_price = float(prices.get(name, 0) or 0)
        subtotal = quantity * unit_price
        rows.append({
            "Product": name,
            "Detected quantity": quantity,
            "Estimated unit price (₹)": unit_price,
            "Estimated subtotal (₹)": subtotal,
        })

    results_df = pd.DataFrame(rows, columns=columns)
    total_estimated_value = results_df["Estimated subtotal (₹)"].sum()

    # =====================================================
    # FINAL DASHBOARD
    # =====================================================

    st.subheader("📊 Final Shop Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Product Value",
            f"₹{total_estimated_value:,.2f}"
        )

    with col2:
        st.metric(
            "Total Detected Instances",
            f"{total_detections:,}"
        )

    with col3:
        st.metric(
            "Distinct Categories Detected",
            len(combined_counts)
        )

    st.caption(
        f"Successfully processed {processed_images} of "
        f"{len(uploaded_files)} uploaded images."
    )

    st.subheader("💰 Estimated Value by Product")

    if not results_df.empty:
        st.dataframe(
            results_df.style.format({
                "Estimated unit price (₹)": "₹{:,.2f}",
                "Estimated subtotal (₹)": "₹{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Product Value Report (CSV)",
            data=csv_data,
            file_name="kirana_shop_value_report.csv",
            mime="text/csv"
        )

    else:
        st.info("No products were detected in the uploaded images.")

    st.warning(
        "IMPORTANT: This is an approximate value based on detected "
        "product instances and editable estimated prices. Products "
        "missed by the model are excluded. The same products may appear "
        "in multiple photos and be counted repeatedly. Therefore, this "
        "is not a verified unique inventory valuation."
    )

elif not uploaded_files:
    st.info("Upload shop photos to start the assessment.")
