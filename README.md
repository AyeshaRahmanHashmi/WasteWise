<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0f0f,100:1a1a1a&height=150&section=header&text=%20WasteWise&fontSize=42&fontColor=00ff9c&animation=fadeIn&fontAlignY=38&desc=Mobile%20App%20Development%20Lab%20Assignment&descAlignY=55&descSize=16" width="100%"/>

<a href="https://git.io/typing-svg"> <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&speed=50&pause=400&center=true&vCenter=true&width=600&color=1E90FF&lines=WasteWise+%E2%80%94+Smart+Waste+Classification;Pick+a+photo%2C+get+the+right+bin;On-device+AI+with+TensorFlow+Lite" alt="Typing SVG" /> </a>

<br>

</div>

<br>

```yaml
project: "WasteWise: On-Device Recycling Bin Guidance"
type: "Native Android app (Java) with TensorFlow Lite"
author: "Ayesha Rahman"
roll_number: "242201006"
class: "BSCS 5A"
course: "Mobile App Development"
model: "MobileNet V1 (recycle_model.tflite)"
```

<br>

![Status](https://img.shields.io/badge/STATUS-Working-2EA043?style=for-the-badge&labelColor=000000)
![Platform](https://img.shields.io/badge/PLATFORM-Android-FFD700?style=for-the-badge&labelColor=000000)
![Offline](https://img.shields.io/badge/INFERENCE-On--Device-FF6B6B?style=for-the-badge&labelColor=000000)

<br>

### 📖 About

Many people are unsure whether a bottle, carton or tin belongs in the plastic, paper or metal bin, and wrongly sorted waste cannot be recycled. **WasteWise** solves this on the phone itself: you pick a photo from the gallery, a TensorFlow Lite image classifier identifies the object, and the app tells you which bin to use, for example *"Item: Plastic Bottle - Dispose in Blue Bin"*.

All processing happens on the device, so the app **works offline** and your photos **never leave the phone**.

<br>

**⚙️ Tech Stack**

![Java](https://img.shields.io/badge/Java-000000?style=for-the-badge&logo=openjdk&logoColor=00FF9C)
![Android](https://img.shields.io/badge/Android-000000?style=for-the-badge&logo=android&logoColor=00FF9C)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-000000?style=for-the-badge&logo=tensorflow&logoColor=00FF9C)
![Gradle](https://img.shields.io/badge/Gradle-000000?style=for-the-badge&logo=gradle&logoColor=00FF9C)
![Git](https://img.shields.io/badge/Git-000000?style=for-the-badge&logo=git&logoColor=00FF9C)
![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=00FF9C)

<br>

### ✨ Features

```bash
$ cat features.txt
-  Gallery image selection (ActivityResultLauncher API)
-  Safe decoding with a software bitmap (no hardware-bitmap crashes)
-  On-device MobileNet V1 classification, no internet needed
-  Model input/output shapes read at run time (float or quantized)
-  Confidence score shown next to the detected item
-  Low-confidence warning asking for a clearer photo
-  Recycling advice: plastic, glass, paper, metal, organic, general
-  Interpreter released in onDestroy to free native memory
```

<br>

### ♻️ Bin Recommendation Rules

The predicted ImageNet label is matched against keywords. 

| Label contains | Result |
|---|---|
| wine bottle, glass jar | Green Bin (glass) |
| bottle, plastic, jug, bucket, bag, container | Orange Bin (plastic) |
| paper, carton, envelope, newspaper, cardboard | Yellow Bin (paper) |
| can, tin, metal, foil | Red Bin (metal) |
| banana, orange, apple, bread, similar food | Grey Bin (organic) |
| anything else | Black Bin (general waste) |

<br>

### 🗂️ Project Structure

```bash
WasteWise/
├── app/
│   ├── build.gradle                  # Module config + TensorFlow Lite dependency
│   └── src/main/
│       ├── AndroidManifest.xml       # App declaration, launcher activity
│       ├── assets/
│       │   ├── recycle_model.tflite  # MobileNet V1 model
│       │   └── labels.txt            # ImageNet labels (1001 lines)
│       ├── java/.../MainActivity.java # Gallery picker, preprocessing, inference, bin logic
│       └── res/
│           └── layout/activity_main.xml # ImageView, Button, result card
├── build.gradle                      # Project-level Gradle config
├── settings.gradle
└── README.md
```

<br>

### 📋 Prerequisites

| Requirement | Details |
|---|---|
| Android Studio | Latest stable version (Hedgehog or newer recommended) |
| JDK | JDK 17 (bundled with recent Android Studio) |
| Android SDK | API 28 or higher (needed for `ImageDecoder`) |
| Git | To clone the repository |
| Device | Physical Android phone with USB debugging **or** an Android emulator (API 28+) |

<br>

### 📦 Dependencies

```gradle

dependencies {
    implementation "org.tensorflow:tensorflow-lite:2.14.0"
    implementation "androidx.appcompat:appcompat:<version>"   // AppCompat
}
```

<br>

### 🚀 Installation and Running

```bash
# 1. Clone the repo
git clone https://github.com/AyeshaRahmanHashmi/WasteWise.git

# 2. Move into the folder
cd WasteWise
```

1. Open **Android Studio** and choose **File → Open**, then select the cloned `WasteWise` folder.
2. Wait for **Gradle sync** to finish (it downloads TensorFlow Lite and other dependencies).
3. Connect a phone with **USB debugging** enabled, or start an emulator from **Device Manager**.
4. Press **Run ▶** to build and install the app.
5. On the phone, open **WasteWise**, tap **Scan Item**, pick a photo from the gallery and read the result card.

**Run from the zipped project:** extract `WasteWise.zip`, then follow steps 1 to 5 above starting from **File → Open**.

**Install the APK directly (optional):** copy `app-debug.apk` to the phone, allow installs from unknown sources and open it.

<br>

### 🔄 How It Works

```text
Gallery photo
   │
   ▼
ImageDecoder (software bitmap, ARGB_8888)
   │
   ▼
Resize to model input (224 x 224) → normalize RGB to [-1, 1] → ByteBuffer
   │
   ▼
TFLite Interpreter  →  float[1][1001] probabilities
   │
   ▼
Best index → label (labels.txt) → keyword rules → bin message + confidence
```

<br>

### 🧪 Test Results

| # | Test image | Expected | Result |
|---|------------|----------|--------|
| 1 | Plastic water bottle | Orange Bin | Orange Bin |
| 2 | Glass bottle | Green Bin | Green Bin |
| 3 | Newspaper or carton | Yellow Bin | Yellow Bin |
| 4 | Metal tin | Red Bin | Red Bin |
| 5 | Apple | Grey Bin | Grey Bin |
| 6 | Unrelated or blurry image | Low-confidence / general waste | Unable to classify |

<br>

### ⚠️ Limitations

MobileNet V1 is trained on ImageNet (general objects, not recycling categories), so the app relies on keyword matching. Items outside the 1000 classes fall into general waste, and lighting, clutter or several objects in one photo reduce accuracy.

<br>

### 🔮 Next Quests

- [ ] Train or fine-tune a model on a dedicated waste dataset
- [ ] Live camera classification with CameraX
- [ ] Quantized model + GPU delegate for faster inference
- [ ] Configurable bin rules per city or country
- [ ] History screen and multilingual support

<br>

### 💌 Let's Connect

Found a bug or have an idea? Feel free to open an issue or reach out.

<div align="center">

<a href="https://www.linkedin.com/in/ayesha-rahman-1060933a2"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" height="45"/></a>
<a href="https://instagram.com/ayesharahman900"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" height="45"/></a>
<a href="mailto:ayesha.rahman.hashmi@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" height="45"/></a>

</div>

---

<div align="center">

![Profile Views](https://komarev.com/ghpvc/?username=AyeshaRahmanHashmi&color=00ff9c&style=flat-square&label=Profile+Views)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a1a,100:0f0f0f&height=100&section=footer" width="100%"/>

</div>
