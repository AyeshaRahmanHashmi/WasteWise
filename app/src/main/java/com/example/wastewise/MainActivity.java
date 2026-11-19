package com.example.wastewise;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "WasteWiseDebug";

    ImageView imageView;
    com.google.android.material.button.MaterialButton buttonPickImage;    TextView textViewResult;
    Interpreter interpreter;
    List<String> labels = new ArrayList<>();

    // Populated dynamically from the model — do NOT hardcode these
    private int imageHeight = 224;
    private int imageWidth = 224;
    private int numClasses = 0;
    private DataType inputDataType = DataType.FLOAT32;
    private boolean isInputQuantized = false;
    private float inputScale = 1.0f;
    private int inputZeroPoint = 0;

    // Minimum confidence below which we flag "unsure"
    private static final float MIN_CONFIDENCE = 0.30f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        buttonPickImage = findViewById(R.id.buttonPickImage);
        textViewResult = findViewById(R.id.textViewResult);

        loadLabels();
        loadModel();

        ActivityResultLauncher<Intent> imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        try {
                            Bitmap bitmap;
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                                ImageDecoder.Source source =
                                        ImageDecoder.createSource(getContentResolver(), imageUri);
                                bitmap = ImageDecoder.decodeBitmap(source, (decoder, info, src) ->
                                        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
                            } else {
                                bitmap = MediaStore.Images.Media.getBitmap(
                                        this.getContentResolver(), imageUri);
                            }
                            imageView.setImageBitmap(bitmap);
                            findViewById(R.id.placeholderIcon).setVisibility(android.view.View.GONE);

                            classifyImage(bitmap);

                        } catch (Exception e) {
                            Log.e(TAG, "Error loading image: " + e.getMessage());
                            Toast.makeText(this, "Failed to load image",
                                    Toast.LENGTH_SHORT).show();
                        }
                    }
                }
        );

        buttonPickImage.setOnClickListener(v -> {
            if (interpreter == null) {
                Toast.makeText(this, "Model not loaded", Toast.LENGTH_SHORT).show();
                return;
            }
            Intent intent = new Intent(Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            imagePickerLauncher.launch(intent);
        });
    }

    // ---------------------------------------------------------------
    //  Model loading + introspection
    // ---------------------------------------------------------------
    private void loadModel() {
        try {
            interpreter = new Interpreter(loadModelFile());

            // ---- INPUT ----
            Tensor inputTensor = interpreter.getInputTensor(0);
            int[] inputShape = inputTensor.shape();
            Log.d(TAG, "Input shape: " + java.util.Arrays.toString(inputShape));

            if (inputShape.length == 4) {
                // [batch, height, width, channels]
                imageHeight = inputShape[1];
                imageWidth = inputShape[2];
                int channels = inputShape[3];
                if (channels != 3) {
                    Log.w(TAG, "Unexpected input channels: " + channels);
                }
            } else {
                Log.w(TAG, "Unexpected input shape rank: " + inputShape.length);
            }

            inputDataType = inputTensor.dataType();
            isInputQuantized = (inputDataType == DataType.UINT8
                    || inputDataType == DataType.INT8);

            if (isInputQuantized) {
                inputScale = inputTensor.quantizationParams().getScale();
                inputZeroPoint = inputTensor.quantizationParams().getZeroPoint();
                Log.d(TAG, "Quantized input — scale=" + inputScale
                        + " zeroPoint=" + inputZeroPoint);
            } else {
                Log.d(TAG, "Float input — will normalize to [-1, 1] (MobileNet convention)");
            }

            // ---- OUTPUT ----
            Tensor outputTensor = interpreter.getOutputTensor(0);
            int[] outputShape = outputTensor.shape();
            Log.d(TAG, "Output shape: " + java.util.Arrays.toString(outputShape));

            numClasses = outputShape[outputShape.length - 1];
            Log.d(TAG, "Detected numClasses from model: " + numClasses);

            // Sanity check labels vs classes
            if (labels.size() != numClasses) {
                Log.w(TAG, "Mismatch: labels=" + labels.size()
                        + " but model classes=" + numClasses
                        + ". Using min of the two.");
            }

        } catch (Exception e) {
            Log.e(TAG, "Error loading model: " + e.getMessage(), e);
            Toast.makeText(this, "Error loading model", Toast.LENGTH_LONG).show();
        }
    }

    // ---------------------------------------------------------------
    //  Inference
    // ---------------------------------------------------------------
    private void classifyImage(Bitmap bitmap) {
        if (interpreter == null) {
            Toast.makeText(this, "Model not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            if (bitmap.getConfig() == Bitmap.Config.HARDWARE) {
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
            }

            // Optional: center-crop to square before resize to reduce distortion
            Bitmap squareBitmap = centerCropSquare(bitmap);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(
                    squareBitmap, imageWidth, imageHeight, true);

            // ---- Build input buffer ----
            ByteBuffer inputBuffer;
            if (isInputQuantized) {
                inputBuffer = ByteBuffer.allocateDirect(imageWidth * imageHeight * 3);
            } else {
                inputBuffer = ByteBuffer.allocateDirect(4 * imageWidth * imageHeight * 3);
            }
            inputBuffer.order(ByteOrder.nativeOrder());
            inputBuffer.rewind();

            int[] intValues = new int[imageWidth * imageHeight];
            resizedBitmap.getPixels(intValues, 0, imageWidth, 0, 0,
                    imageWidth, imageHeight);

            for (int val : intValues) {
                int r = (val >> 16) & 0xFF;
                int g = (val >> 8) & 0xFF;
                int b = val & 0xFF;

                if (isInputQuantized) {
                    inputBuffer.put((byte) (((r / 255.0f) / inputScale) + inputZeroPoint));
                    inputBuffer.put((byte) (((g / 255.0f) / inputScale) + inputZeroPoint));
                    inputBuffer.put((byte) (((b / 255.0f) / inputScale) + inputZeroPoint));
                } else {
                    // MobileNet convention: [-1, 1]
                    inputBuffer.putFloat((r / 127.5f) - 1.0f);
                    inputBuffer.putFloat((g / 127.5f) - 1.0f);
                    inputBuffer.putFloat((b / 127.5f) - 1.0f);
                }
            }
            inputBuffer.rewind();

            // ---- Allocate output ----
            float[][] outputArray = new float[1][numClasses];
            interpreter.run(inputBuffer, outputArray);

            float[] probabilities = outputArray[0];
            int maxIndex = 0;
            float maxProb = -Float.MAX_VALUE;

            for (int i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }

            String predictedLabel = "Unknown";
            if (maxIndex >= 0 && maxIndex < labels.size()) {
                predictedLabel = labels.get(maxIndex).trim();
            }

            // Low-confidence guard
            if (maxProb < MIN_CONFIDENCE) {
                textViewResult.setText(
                        "⚠️ Not confident enough (" +
                                String.format("%.1f", maxProb * 100) + "%)\n\n" +
                                "Best guess: " + predictedLabel + "\n" +
                                "Try a clearer photo.");
                return;
            }

            String finalResult = getRecyclingInstructions(predictedLabel, maxProb);
            textViewResult.setText(finalResult);

        } catch (Exception e) {
            Log.e(TAG, "Error during inference: " + e.getMessage(), e);
            Toast.makeText(this, "AI Error: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
            textViewResult.setText("Error processing image. Check Logcat.");
        }
    }

    /** Center-crop a bitmap to a square — keeps aspect ratio, avoids stretch artifacts. */
    private Bitmap centerCropSquare(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        int size = Math.min(w, h);
        int x = (w - size) / 2;
        int y = (h - size) / 2;
        return Bitmap.createBitmap(src, x, y, size, size);
    }

    // ---------------------------------------------------------------
    //  Recycling rules
    // ---------------------------------------------------------------
    private String getRecyclingInstructions(String label, float confidence) {
        String lower = label.toLowerCase();
        String binColor = "Unknown";
        String binName = "General Waste";
        String notes = "Check local guidelines.";

        // --- YELLOW BIN: Paper and packaging materials ---
        if (lower.contains("carton") || lower.contains("notebook") || lower.contains("book")
                || lower.contains("envelope") || lower.contains("comic book") || lower.contains("crossword")
                || lower.contains("menu") || lower.contains("packet") || lower.contains("paper towel")
                || lower.contains("toilet tissue")) {
            binColor = "Yellow";
            binName = "Paper and Packaging";
            notes = "Cardboard, newspapers, magazines, etc.";
        }

        // --- GREEN BIN: Glass bottles and glass waste ---
        else if (lower.contains("wine bottle") || lower.contains("beer bottle")
                || lower.contains("water jug") || lower.contains("whiskey jug")
                || lower.contains("jar") || lower.contains("goblet")) {
            binColor = "Green";
            binName = "Glass Waste";
            notes = "Glass jars, wine bottles, etc.";
        }

        // --- GRAY BIN: Food and organic waste ---
        else if (lower.contains("banana") || lower.contains("orange") || lower.contains("lemon")
                || lower.contains("fig") || lower.contains("pineapple") || lower.contains("jackfruit")
                || lower.contains("custard apple") || lower.contains("pomegranate") || lower.contains("strawberry")
                || lower.contains("broccoli") || lower.contains("cauliflower") || lower.contains("cucumber")
                || lower.contains("mushroom") || lower.contains("cabbage") || lower.contains("zucchini")
                || lower.contains("artichoke") || lower.contains("bell pepper") || lower.contains("corn")
                || lower.contains("pizza") || lower.contains("hotdog") || lower.contains("cheeseburger")
                || lower.contains("burrito") || lower.contains("potpie") || lower.contains("guacamole")
                || lower.contains("consomme") || lower.contains("hot pot") || lower.contains("trifle")
                || lower.contains("ice cream") || lower.contains("ice lolly") || lower.contains("French loaf")
                || lower.contains("bagel") || lower.contains("pretzel") || lower.contains("mashed potato")
                || lower.contains("meat loaf") || lower.contains("chocolate sauce") || lower.contains("dough")
                || lower.contains("espresso") || lower.contains("cup") || lower.contains("eggnog")) {
            binColor = "Gray";
            binName = "Food and Organic Waste";
            notes = "Fruit peels, vegetables, leftovers, rotten food.";
        }

        // --- ORANGE BIN: Plastic waste ---
        else if (lower.contains("plastic bag") || lower.contains("pop bottle") || lower.contains("water bottle")
                || lower.contains("pill bottle") || lower.contains("soap dispenser") || lower.contains("lotion")) {
            binColor = "Orange";
            binName = "Plastic Waste";
            notes = "Plastic bottles, bags, containers, etc.";
        }

        // --- RED BIN: Iron and metal waste ---
        else if (lower.contains("can opener") || lower.contains("caldron") || lower.contains("coffeepot")
                || lower.contains("Dutch oven") || lower.contains("frying pan") || lower.contains("iron")
                || lower.contains("ladle") || lower.contains("milk can") || lower.contains("pot")
                || lower.contains("spatula") || lower.contains("strainer") || lower.contains("teapot")
                || lower.contains("wok") || lower.contains("wooden spoon") || lower.contains("chain")
                || lower.contains("hammer") || lower.contains("hatchet") || lower.contains("nail")
                || lower.contains("screw") || lower.contains("screwdriver") || lower.contains("safety pin")
                || lower.contains("shovel") || lower.contains("lock")) {
            binColor = "Red";
            binName = "Metal Waste";
            notes = "Cans, tin cans, scrap iron, etc.";
        }

        // --- PURPLE/GRAY BIN: Leaves and garden waste ---
        else if (lower.contains("daisy") || lower.contains("yellow lady's slipper")
                || lower.contains("rapeseed") || lower.contains("acorn") || lower.contains("hip")
                || lower.contains("buckeye") || lower.contains("coral fungus") || lower.contains("agaric")
                || lower.contains("gyromitra") || lower.contains("stinkhorn") || lower.contains("earthstar")
                || lower.contains("hen-of-the-woods") || lower.contains("bolete") || lower.contains("hay")) {
            binColor = "Purple/Gray";
            binName = "Leaves and Garden Waste";
            notes = "Can be merged with gray bins.";
        }

        // --- E-WASTE (Not in image, but good to have) ---
        else if (lower.contains("cellular telephone") || lower.contains("laptop") || lower.contains("computer") || lower.contains("iPod") || lower.contains("monitor") || lower.contains("remote control")) {
            binColor = "Special Drop-off";
            binName = "E-Waste";
            notes = "Do not put in regular bins. Take to an e-waste facility.";
        }

        // --- CATCH ALL ---
        else {
            binColor = "Black";
            binName = "General Waste";
            notes = "This item does not match a specific recycling category.";
        }

        // Returning the formatted text
        return "Item detected: " + label + "\n\n" +
                "Dispose in: " + binName + " (" + binColor + " Bin)\n\n" +
                "Details: " + notes + "\n\n" +
                "Confidence: " + String.format("%.1f", confidence * 100) + "%";
    }

    // ---------------------------------------------------------------
    //  File I/O
    // ---------------------------------------------------------------
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor afd = getAssets().openFd("recycle_model.tflite");
        try (FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            FileChannel channel = fis.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY,
                    afd.getStartOffset(), afd.getLength());
        }
    }

    private void loadLabels() {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    labels.add(line);
                }
            }
            Log.d(TAG, "Labels loaded: " + labels.size());
        } catch (IOException e) {
            Log.e(TAG, "Error loading labels: " + e.getMessage(), e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}