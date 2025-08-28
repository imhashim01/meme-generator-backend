import os, io, base64
from typing import Tuple
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from PIL import Image,ImageOps, ImageDraw, ImageFont, ImageEnhance, ImageFilter


# ----------------------
# Setup
# ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)

# ----------------------
# Helpers
# ----------------------
def file_to_data_url(file_storage) -> Tuple[str, bytes]:
    raw = file_storage.read()
    file_storage.stream.seek(0)
    mime = file_storage.mimetype or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}", raw

def apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
    """Apply filters and fun overlays (dog ears, sunglasses, flower crown)."""
    filter_name = filter_name.lower().strip()

    if filter_name == "grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    elif filter_name == "sepia":
        sepia = ImageOps.colorize(ImageOps.grayscale(img), "#704214", "#C0A080")
        return sepia.convert("RGB")
    elif filter_name == "blur":
        return img.filter(ImageFilter.GaussianBlur(3))
    elif filter_name == "bright":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.4)
    elif filter_name == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.8)

    # 👇 Fun overlays from static/filters
    overlay_path = os.path.join("static", "filters", f"{filter_name}.png")
    if os.path.exists(overlay_path):
        overlay = Image.open(overlay_path).convert("RGBA")

        # Resize overlay to ~50% of image width
        W, H = img.size
        scale = W // 2
        aspect = overlay.size[1] / overlay.size[0]
        overlay = overlay.resize((scale, int(scale * aspect)), Image.LANCZOS)

        # Positioning based on filter
        if filter_name == "dog":
            pos = (W // 5, H // 25)   # ears at top center
        elif filter_name == "sunglasses":
            pos = (W // 5, H // 4)  # roughly eyes area
        elif filter_name == "flower":
            pos = (W // 5, 0)   # crown top
        else:
            pos = (W // 2 - overlay.size[0] // 2, H // 2 - overlay.size[1] // 2)

        # Paste with alpha transparency
        img = img.convert("RGBA")
        img.paste(overlay, pos, overlay)
        return img.convert("RGB")

    return img

def draw_caption_on_image(image: Image.Image, text: str, position: str = "bottom") -> bytes:
    """Overlay meme text on the image and return JPEG bytes."""
    draw = ImageDraw.Draw(image)
    W, H = image.size

    base_size = max(24, int(W / 12))
    font = None
    for candidate in ["Impact.ttf", "impact.ttf", "Arial.ttf", "arial.ttf"]:
        try:
            font = ImageFont.truetype(candidate, base_size)
            break
        except Exception:
            continue
    if not font:
        font = ImageFont.load_default()

    # Wrap text
    def wrap_text(txt):
        words, lines, current = txt.split(), [], ""
        for w in words:
            trial = (current + " " + w).strip()
            bbox = draw.textbbox((0, 0), trial, font=font)
            tw = bbox[2] - bbox[0]
            if tw <= int(W * 0.9):
                current = trial
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    lines = wrap_text(text.upper())

    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
    stroke = max(2, int(base_size / 12))
    gap = int(line_height * 0.25)
    total_h = len(lines) * line_height + (len(lines) - 1) * gap

    if position == "top":
        y = int(H * 0.05)
    elif position == "center":
        y = int((H - total_h) / 2)
    else:
        y = H - total_h - int(H * 0.05)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = int((W - tw) / 2)
        draw.text((x, y), line, font=font, fill="white",
                  stroke_width=stroke, stroke_fill="black")
        y += line_height + gap

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    return out.getvalue()


def json_error(message: str, code: int = 400):
    return jsonify({"error": message}), code

# ----------------------
# Routes
# ----------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/api/generate-captions", methods=["POST"])
def generate_captions():
    """
    Accepts: image
    Returns: {captions: [], hashtags: [], description: ""}
    """
    if "image" not in request.files:
        return json_error("No image provided", 400)

    try:
        data_url, _ = file_to_data_url(request.files["image"])

        # Ask OpenAI for funny captions + hashtags
        prompt = f"Generate 3 funny meme captions, 5 trending hashtags, and a short description for this image."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a meme caption generator."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            max_tokens=300,
        )

        text = resp.choices[0].message.content
        captions, hashtags, description = [], [], ""

        if text:
            parts = text.split("\n")
            for line in parts:
                line = line.strip("-• ")
                if not line:
                    continue
                if line.startswith("#"):
                    hashtags.append(line.strip("#"))
                elif len(captions) < 3:
                    captions.append(line)
                else:
                    description += line + " "

        return jsonify({
            "captions": captions,
            "hashtags": hashtags,
            "description": description.strip()
        })

    except Exception as e:
        print("❌ /api/generate-captions error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/finalize-meme", methods=["POST"])
def finalize_meme():
    try:
        if "image" not in request.files:
            return json_error("No image provided", 400)

        text = (request.form.get("text") or "").strip()
        position = (request.form.get("position") or "bottom").strip().lower()
        filter_name = (request.form.get("filter") or "").strip().lower()

        if not text:
            return json_error("Missing 'text' field.", 400)

        # Load uploaded image
        _, raw = file_to_data_url(request.files["image"])
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        # Apply filter first
        image = apply_filter(image, filter_name)

        # Save filtered image into bytes (valid JPEG)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        raw_filtered = buf.getvalue()

        # Draw caption on top
        final_bytes = draw_caption_on_image(
            Image.open(io.BytesIO(raw_filtered)), text, position=position
        )

        return jsonify({
            "image_base64": base64.b64encode(final_bytes).decode("utf-8")
        })

    except Exception as e:
        print("❌ /api/finalize-meme error:", e)
        return jsonify({"error": str(e)}), 500

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
