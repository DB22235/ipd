# Rice Disease Mobile Preprocessing & Inference Specification

**Version:** 1.0 (Mobile Prototype)  
**Target Architecture:** MobileNetV3-Large LiteRT / TFLite  
**Target Hardware:** Android (Java/Kotlin / NDK) & iOS (Swift / CoreML / TFLite)  

---

## 1. Input Specification Contract

| Parameter | Value / Contract |
| :--- | :--- |
| **Tensor Dimensions** | `[1, 224, 224, 3]` (Batch, Height, Width, Channels) |
| **Data Type** | `float32` (Float32 values in memory) |
| **Color Space** | **RGB** (Red, Green, Blue) |
| **Pixel Value Range** | **`[0.0, 255.0]` (Raw unscaled float32)** |
| **Neutral Fill Canvas** | `RGB(114, 114, 114)` |
| **Resize Strategy** | **Aspect-preserving letterboxing** (Bicubic or Bilinear) |

> [!CAUTION]
> **Common Mobile Integration Traps:**
> 1. **Do NOT divide by 255.0:** The model's first internal layer is an explicit Keras `Rescaling(1./255)` layer. Passing inputs scaled in `[0.0, 1.0]` will result in double-normalization and catastrophic misclassifications!
> 2. **Ensure RGB Order:** Android `Bitmap` pixels are often decoded as ARGB or BGR. Ensure channels are strictly ordered as RGB before feeding the tensor buffer.
> 3. **Never squish or stretch the aspect ratio:** Rice leaves have elongated aspect ratios. Center the image within a $224 \times 224$ canvas using neutral padding `(114, 114, 114)`.

---

## 2. Pre-Inference Quality & Botanical Gate (Safe Abstention)

Before running neural network inference, the mobile client must evaluate two lightweight quality checks:

### Step 2.1: Botanical Foliage Area Check
Verify that a green plant leaf is actually visible in the camera frame:
- Convert camera frame to HSV color space.
- Green foliage mask: $\text{Hue} \in [25, 90]$, $\text{Sat} \ge 35$, $\text{Val} \ge 35$.
- Compute foliage ratio: $\text{ratio} = \frac{\text{green\_pixels}}{\text{total\_pixels}}$.
- **Gate:** If $\text{ratio} < 0.05$ (less than 5% of the frame is foliage):
  - **Return:** `unsupported_input`
  - **User Feedback:** *"No foliage detected. Please center a rice leaf in the camera frame."*

### Step 2.2: Extreme Blur Detection
Verify that the camera is focused on the leaf surface:
- Compute variance of the Laplacian filter on the grayscale image: $\sigma^2_{\text{Laplacian}}$.
- **Gate:** If $\sigma^2_{\text{Laplacian}} < 40.0$:
  - **Return:** `unsupported_input`
  - **User Feedback:** *"Image is too blurry. Tap the screen to focus on the leaf lesion."*

---

## 3. Aspect-Preserving Letterbox Algorithm (Pseudocode)

```kotlin
fun preprocessImage(bitmap: Bitmap, targetSize: Int = 224): ByteBuffer {
    val origW = bitmap.width.toFloat()
    val origH = bitmap.height.toFloat()
    val scale = minOf(targetSize / origW, targetSize / origH)
    val newW = (origW * scale).roundToInt()
    val newH = (origH * scale).roundToInt()

    // 1. Resize leaf maintaining aspect ratio
    val scaledBitmap = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

    // 2. Create target canvas with neutral gray fill (114, 114, 114)
    val canvasBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(canvasBitmap)
    canvas.drawColor(Color.rgb(114, 114, 114))

    val padX = ((targetSize - newW) / 2).toFloat()
    val padY = ((targetSize - newH) / 2).toFloat()
    canvas.drawBitmap(scaledBitmap, padX, padY, null)

    // 3. Populate Float32 ByteBuffer in raw [0.0, 255.0] range (RGB)
    val buffer = ByteBuffer.allocateDirect(1 * targetSize * targetSize * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    
    val pixels = IntArray(targetSize * targetSize)
    canvasBitmap.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

    for (pixel in pixels) {
        val r = ((pixel shr 16) and 0xFF).toFloat() // [0.0, 255.0]
        val g = ((pixel shr 8) and 0xFF).toFloat()  // [0.0, 255.0]
        val b = (pixel and 0xFF).toFloat()         // [0.0, 255.0]
        buffer.putFloat(r)
        buffer.putFloat(g)
        buffer.putFloat(b)
    }
    buffer.rewind()
    return buffer
}
```

---

## 4. Output Specification & Confidence Gating

### Output Tensor Format
- Shape: `[1, 4]`
- Type: `float32` Softmax probabilities summing to $1.0$.
- Authoritative Class Index Mapping:
  - `Index 0:` **`blast`**
  - `Index 1:` **`blight`**
  - `Index 2:` **`brown_spot`**
  - `Index 3:` **`healthy`**

### Post-Inference Decision Gating
Let $p_{(1)}$ be the highest probability and $p_{(2)}$ be the second highest:
1. **Low Confidence Gate:** If $p_{(1)} < 0.60$:
   - **Return:** `uncertain`
   - **User Feedback:** *"Low confidence diagnosis. Retake photo under better lighting."*
2. **Ambiguous Margin Gate:** If $(p_{(1)} - p_{(2)}) < 0.20$:
   - **Return:** `uncertain`
   - **User Feedback:** *"Ambiguous lesion features between two conditions. Retake closer to the lesion."*
3. **Confirmed Diagnosis:** If $p_{(1)} \ge 0.60$ and $(p_{(1)} - p_{(2)}) \ge 0.20$:
   - **Return:** Class name at $\arg\max_i(p_i)$ with confidence percentage.
