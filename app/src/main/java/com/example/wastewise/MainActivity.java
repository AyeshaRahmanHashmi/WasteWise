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

import org.tensorflow.lite.Interpreter;

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

    private static final String TAG = "RecycleRightDebug";
    ImageView imageView;
    Button buttonPickImage;
    TextView textViewResult;
    Interpreter interpreter;
    List<String> labels = new ArrayList<>();

    private int imageSize = 224; // Default, but we will read from model
    private static final int NUM_CLASSES = 1000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        buttonPickImage = findViewById(R.id.buttonPickImage);
        textViewResult = findViewById(R.id.textViewResult);

        loadLabels();

        try {
            interpreter = new Interpreter(loadModelFile());

            // CRASH PREVENTION: Check what size the model actually expects
            int[] inputShape = interpreter.getInputTensor(0).shape();
            if (inputShape.length >= 3) {
                imageSize = inputShape[1]; // Usually 224
                Log.d(TAG, "Model expects image size: " + imageSize);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error loading model: " + e.getMessage());
            Toast.makeText(this, "Error loading model", Toast.LENGTH_LONG).show();
        }

        ActivityResultLauncher<Intent> imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        try {
                            // SAFER IMAGE LOADING
                            Bitmap bitmap;
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                                ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), imageUri);
                                // FORCE SOFTWARE ALLOCATION SO WE CAN READ PIXELS
                                bitmap = ImageDecoder.decodeBitmap(source, (decoder, info, src) -> {
                                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                                });
                            } else {
                                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                            }
                            imageView.setImageBitmap(bitmap);
                            classifyImage(bitmap);

                        } catch (Exception e) {
                            Log.e(TAG, "Error loading image: " + e.getMessage());
                            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                        }
                    }
                }
        );

        buttonPickImage.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            imagePickerLauncher.launch(intent);
        });
    }

    private void classifyImage(Bitmap bitmap) {
        try {
            // Ensure the bitmap is not hardware accelerated before reading pixels
            if (bitmap.getConfig() == Bitmap.Config.HARDWARE) {
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
            }

            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            inputBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

            inputBuffer.rewind();
            for (int i = 0; i < imageSize * imageSize; i++) {
                final int val = intValues[i];
                float r = ((val >> 16) & 0xFF) / 127.5f - 1.0f;
                float g = ((val >> 8) & 0xFF) / 127.5f - 1.0f;
                float b = (val & 0xFF) / 127.5f - 1.0f;

                inputBuffer.putFloat(r);
                inputBuffer.putFloat(g);
                inputBuffer.putFloat(b);
            }

            float[][] outputArray = new float[1][NUM_CLASSES];
            interpreter.run(inputBuffer, outputArray);

            float[] probabilities = outputArray[0];
            int maxIndex = 0;
            float maxProb = 0.0f;

            for (int i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }

            String predictedLabel = "Unknown";
            if (maxIndex >= 0 && maxIndex < labels.size()) {
                predictedLabel = labels.get(maxIndex);
            }

            String finalResult = getRecyclingInstructions(predictedLabel, maxProb);
            textViewResult.setText(finalResult);

        } catch (Exception e) {
            Log.e(TAG, "Error during inference: " + e.getMessage());
            Toast.makeText(this, "AI Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            textViewResult.setText("Error processing image. Check Logcat.");
        }
    }

    private String getRecyclingInstructions(String label, float confidence) {
        String lowerCaseLabel = label.toLowerCase();
        String binType = "General Waste (Black Bin)";

        if (lowerCaseLabel.contains("bottle") || lowerCaseLabel.contains("plastic") || lowerCaseLabel.contains("bag")) {
            binType = "Recycle (Blue Bin)";
        } else if (lowerCaseLabel.contains("paper") || lowerCaseLabel.contains("book") || lowerCaseLabel.contains("cardboard")) {
            binType = "Recycle (Green Bin)";
        } else if (lowerCaseLabel.contains("can") || lowerCaseLabel.contains("tin") || lowerCaseLabel.contains("metal")) {
            binType = "Recycle (Yellow Bin)";
        } else if (lowerCaseLabel.contains("glass") || lowerCaseLabel.contains("wine")) {
            binType = "Recycle (White Bin)";
        }

        return "Item detected: " + label + "\n\n" +
                "Dispose in: " + binType + "\n\n" +
                "Confidence: " + String.format("%.1f", (confidence * 100)) + "%";
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("recycle_model.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }

    private void loadLabels() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
            reader.close();
            Log.d(TAG, "Labels loaded: " + labels.size());
        } catch (IOException e) {
            Log.e(TAG, "Error loading labels: " + e.getMessage());
        }
    }
}