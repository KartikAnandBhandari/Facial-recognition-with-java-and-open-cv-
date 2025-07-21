package com.example;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AttendanceSystem extends JFrame {
    static {
        String dllPath = System.getenv("OPENCV_DLL_PATH") != null
                ? System.getenv("OPENCV_DLL_PATH")
                : System.getProperty("user.dir") + "/lib/opencv_java490.dll";
        try {
            System.load(dllPath);
            System.out.println("OpenCV DLL loaded successfully from: " + dllPath);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV DLL: " + e.getMessage());
            System.exit(1);
        }
    }

    private static final Logger logger = LoggerFactory.getLogger(AttendanceSystem.class);
    private final JLabel cameraLabel = new JLabel();
    private VideoCapture capture;
    private final Map<Integer, String> employeeMap = new HashMap<>();
    private final Map<Integer, List<double[]>> knownFaceEncodings = new HashMap<>();
    private final Set<Integer> markedToday = new HashSet<>();
    private final JButton startButton = new JButton("Start");
    private final JButton stopButton = new JButton("Stop");
    private final String ATTENDANCE_FILE = "attendance.txt";
    private volatile boolean isRunning;
    private final AtomicBoolean unknownPopupActive = new AtomicBoolean(false);
    private final AtomicBoolean attendancePopupActive = new AtomicBoolean(false);
    private long lastUnknownPopupTime = 0;
    private boolean hasShownUnknownPopup = false;
    private Net recognitionNet;
    private Net detectionNet;
    private final ExecutorService executor = Executors.newFixedThreadPool(6);
    private static final long UNKNOWN_POPUP_COOLDOWN_MS = 30000;
    private static final double MATCH_THRESHOLD = 0.4;
    private static final double DETECTION_CONFIDENCE = 0.4;
    private static final int MIN_FACE_SIZE = 50;

    public AttendanceSystem() {
        loadRecognitionModel();
        loadDetectionModel();
        loadEmployeeData();
        setupUI();

        setTitle("Facial Recognition Attendance System");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLocationRelativeTo(null);
    }

    private String getModelPath(String modelName) {
        return new File(System.getProperty("user.dir") + "/src/resources/models/" + modelName).getAbsolutePath();
    }

    private void loadRecognitionModel() {
        try {
            recognitionNet = Dnn.readNetFromONNX(getModelPath("face_recognition_sface_2021dec.onnx"));
            if (recognitionNet.empty()) throw new IOException("Failed to load recognition model");
            logger.info("Recognition model loaded successfully");
        } catch (IOException | RuntimeException e) {
            logger.error("Error loading recognition model: " + e.getMessage(), e);
            showPopup("❌ Error loading recognition model: " + e.getMessage());
        }
    }

    private void loadDetectionModel() {
        try {
            detectionNet = Dnn.readNetFromCaffe(
                getModelPath("deploy.prototxt"),
                getModelPath("res10_300x300_ssd_iter_140000.caffemodel")
            );
            if (detectionNet.empty()) throw new IOException("Failed to load detection model");
            logger.info("Detection model loaded successfully");
        } catch (IOException | RuntimeException e) {
            logger.error("Error loading detection model: " + e.getMessage(), e);
            showPopup("❌ Error loading detection model: " + e.getMessage());
        }
    }

    private void loadEmployeeData() {
        File file = new File("employees.txt");
        if (!file.exists()) {
            logger.info("No employee data file found, starting with empty employee map");
            return;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 2) continue;

                int id = Integer.parseInt(parts[0]);
                String name = parts[1].trim();
                employeeMap.put(id, name);

                List<double[]> encodings = new ArrayList<>();
                for (int i = 0; i < 5; i++) {
                    String path = "encodings/face_" + id + "_" + i + ".vec";
                    File vecFile = new File(path);
                    if (vecFile.exists()) {
                        double[] enc = loadEncodingFromFile(path);
                        if (enc != null) encodings.add(enc);
                    }
                }

                if (!encodings.isEmpty()) {
                    knownFaceEncodings.put(id, encodings);
                    logger.info("Loaded encodings for employee ID: " + id);
                }
            }
        } catch (IOException e) {
            logger.error("Error loading employee data: " + e.getMessage(), e);
            showPopup("❌ Error loading employee data: " + e.getMessage());
        }
    }

    private void setupUI() {
        setLayout(new BorderLayout());
        add(cameraLabel, BorderLayout.CENTER);

        JPanel controlPanel = new JPanel();
        JButton registerButton = new JButton("Register New Employee");

        startButton.addActionListener(e -> executor.submit(this::startCamera));
        stopButton.addActionListener(e -> executor.submit(this::stopCamera));
        registerButton.addActionListener(e -> executor.submit(() -> {
            stopCamera();
            registerNewEmployee();
            startCamera();
        }));

        stopButton.setEnabled(false);
        controlPanel.add(startButton);
        controlPanel.add(stopButton);
        controlPanel.add(registerButton);
        add(controlPanel, BorderLayout.SOUTH);

        setVisible(true);
        logger.info("UI setup complete");
    }

    private void startCamera() {
    if (isRunning) {
        logger.info("Camera already running");
        return;
    }

    int[] indices = {0, 1};
    VideoCapture tempCapture = null;

    for (int index : indices) {
        int[] backends = {Videoio.CAP_DSHOW, Videoio.CAP_MSMF, Videoio.CAP_ANY};
        for (int backend : backends) {
            logger.info("Trying camera index: " + index + ", backend: " + backend);
            tempCapture = new VideoCapture(index, backend);
            if (tempCapture.isOpened()) {
                tempCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
                tempCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
                logger.info("Camera opened with index: " + index + ", backend: " + backend +
                        ", resolution=" + tempCapture.get(Videoio.CAP_PROP_FRAME_WIDTH) + "x" + tempCapture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
                capture = tempCapture;
                break;
            }
        }
        if (capture != null && capture.isOpened()) break;
    }

    if (capture == null || !capture.isOpened()) {
        logger.error("❌ Failed to open camera.");
        showPopup("❌ Failed to open camera. Check drivers and permissions.");
        if (tempCapture != null) tempCapture.release();
        return;
    }

    try {
        Thread.sleep(1000);  // Warm-up
        logger.info("Camera warm-up complete");
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }

    isRunning = true;

    executor.submit(() -> {
        Mat frame = new Mat();

        try {
            while (isRunning && capture.isOpened()) {
                if (!capture.read(frame) || frame.empty()) {
                    logger.warn("⚠️ Frame capture failed or returned empty.");
                    Thread.sleep(200);
                    continue;
                }

                logger.info("✅ First valid frame captured: size=" + frame.size());

                Mat processedFrame = frame.clone();
                detectAndRecognizeFaces(processedFrame);

                final Mat finalDisplayFrame = processedFrame.clone(); // Preserve processed image for UI
                SwingUtilities.invokeLater(() -> {
                    if (finalDisplayFrame.empty() || finalDisplayFrame.width() == 0 || finalDisplayFrame.height() == 0) {
                        logger.error("🚫 Empty image passed to matToImage(), size: " + finalDisplayFrame.size());
                        finalDisplayFrame.release();
                        return;
                    }

                    ImageIcon icon = matToImage(finalDisplayFrame);
                    if (icon != null) {
                        cameraLabel.setIcon(icon);
                    } else {
                        logger.error("matToImage returned null for frame size: " + finalDisplayFrame.size());
                    }
                    finalDisplayFrame.release();
                });

                processedFrame.release();
            }
        } catch (Exception e) {
            logger.error("💥 Error in camera loop: " + e.getMessage(), e);
            showPopup("💥 Camera error: " + e.getMessage());
        } finally {
            frame.release();
            stopCamera();
        }
    });

    SwingUtilities.invokeLater(() -> {
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
    });
}


    private void stopCamera() {
        isRunning = false;
        if (capture != null && capture.isOpened()) {
            capture.release();
            capture = null;
            logger.info("Camera released");
        }

        SwingUtilities.invokeLater(() -> {
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            cameraLabel.setIcon(null);
            logger.info("Camera UI reset");
        });
    }

    private void registerNewEmployee() {
        String name = JOptionPane.showInputDialog("Enter employee name:");
        if (name == null || name.trim().isEmpty()) {
            logger.warn("Employee registration cancelled or invalid name");
            return;
        }
        name = name.trim();

        int newId = employeeMap.keySet().stream().mapToInt(i -> i).max().orElse(0) + 1;
        List<double[]> encodings = new ArrayList<>();
        Mat savedFrame = null;

        VideoCapture cap = new VideoCapture(0, Videoio.CAP_ANY);
        if (!cap.isOpened()) {
            logger.error("Camera failed to open for registration");
            showPopup("❌ Camera failed to open for registration");
            return;
        }

        JFrame preview = new JFrame("Capturing [0/3]");
        JLabel label = new JLabel();
        preview.add(label);
        preview.setSize(400, 300);
        preview.setLocationRelativeTo(null);
        preview.setVisible(true);

        int tries = 0;
        while (encodings.size() < 3 && tries < 50) {
            Mat frame = new Mat();
            if (!cap.read(frame) || frame.empty()) {
                logger.warn("Failed to capture frame during registration, try: " + tries);
                frame.release();
                tries++;
                continue;
            }

            Mat inputBlob = Dnn.blobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104, 117, 123));
            detectionNet.setInput(inputBlob);
            Mat detections = detectionNet.forward().reshape(1, (int) detectionNet.forward().total() / 7);

            Rect bestRect = null;
            double maxConfidence = 0;

            for (int i = 0; i < detections.rows(); i++) {
                double conf = detections.get(i, 2)[0];
                if (conf > DETECTION_CONFIDENCE && conf > maxConfidence) {
                    int x1 = (int) (detections.get(i, 3)[0] * frame.cols());
                    int y1 = (int) (detections.get(i, 4)[0] * frame.rows());
                    int x2 = (int) (detections.get(i, 5)[0] * frame.cols());
                    int y2 = (int) (detections.get(i, 6)[0] * frame.rows());
                    if ((x2 - x1) >= MIN_FACE_SIZE && (y2 - y1) >= MIN_FACE_SIZE) {
                        bestRect = new Rect(x1, y1, x2 - x1, y2 - y1);
                        maxConfidence = conf;
                    }
                }
            }

            if (bestRect != null) {
                Mat faceROI = new Mat(frame, bestRect);
                double[] encoding = generateFaceEncoding(faceROI);
                if (encoding != null) {
                    encodings.add(encoding);
                    if (savedFrame == null) savedFrame = frame.clone();
                    logger.info("Captured encoding " + encodings.size() + " for employee ID: " + newId);
                } else {
                    logger.warn("Failed to generate encoding for face in frame, try: " + tries);
                }
                faceROI.release();
            }

            SwingUtilities.invokeLater(() -> {
                ImageIcon icon = matToImage(frame);
                if (icon != null) {
                    label.setIcon(icon);
                    preview.setTitle("Capturing [" + encodings.size() + "/3]");
                } else {
                    logger.error("matToImage returned null during registration, frame size: " + frame.size());
                }
            });

            frame.release();
            tries++;
            try {
                Thread.sleep(100); // Stabilize frame capture
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.error("Registration thread interrupted: " + e.getMessage());
            }
        }

        cap.release();
        preview.dispose();

        if (encodings.size() < 2) {
            logger.error("Not enough valid encodings for employee ID: " + newId);
            showPopup("❌ Not enough valid encodings");
            if (savedFrame != null) savedFrame.release();
            return;
        }

        try {
            Imgcodecs.imwrite("faces/face_" + newId + ".jpg", savedFrame);
            logger.info("Saved face image for employee ID: " + newId);
        } catch (Exception e) {
            logger.error("Error saving face image for employee ID: " + newId + ": " + e.getMessage());
        } finally {
            if (savedFrame != null) savedFrame.release();
        }

        for (int i = 0; i < encodings.size(); i++) {
            saveEncodingToFile("encodings/face_" + newId + "_" + i + ".vec", encodings.get(i));
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("employees.txt", true))) {
            writer.write(newId + "," + name);
            writer.newLine();
            logger.info("Saved employee data: ID=" + newId + ", Name=" + name);
        } catch (IOException e) {
            logger.error("Error saving employee data: " + e.getMessage(), e);
            showPopup("❌ Error saving employee data: " + e.getMessage());
        }

        employeeMap.put(newId, name);
        knownFaceEncodings.put(newId, encodings);
        showPopup("✅ Registered: " + name);
        logger.info("Employee registered successfully: ID=" + newId + ", Name=" + name);
    }

    private void detectAndRecognizeFaces(Mat frame) {
        if (frame.empty()) {
            logger.error("Input frame to detectAndRecognizeFaces is empty");
            return;
        }
        logger.debug("Processing frame: size=" + frame.size() + ", channels=" + frame.channels());

        Mat inputBlob = Dnn.blobFromImage(frame, 1.0, new Size(300, 300), new Scalar(104, 117, 123));
        detectionNet.setInput(inputBlob);
        Mat detections = detectionNet.forward().reshape(1, (int) detectionNet.forward().total() / 7);

        for (int i = 0; i < detections.rows(); i++) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > DETECTION_CONFIDENCE) {
                int x1 = (int) (detections.get(i, 3)[0] * frame.cols());
                int y1 = (int) (detections.get(i, 4)[0] * frame.rows());
                int x2 = (int) (detections.get(i, 5)[0] * frame.cols());
                int y2 = (int) (detections.get(i, 6)[0] * frame.rows());

                if ((x2 - x1) < MIN_FACE_SIZE || (y2 - y1) < MIN_FACE_SIZE) {
                    logger.debug("Face too small: width=" + (x2 - x1) + ", height=" + (y2 - y1));
                    continue;
                }

                Rect faceRect = new Rect(new Point(x1, y1), new Point(x2, y2));
                Mat faceROI = new Mat(frame, faceRect);
                double[] encoding = generateFaceEncoding(faceROI);

                int bestId = -1;
                double bestScore = -1;

                if (encoding != null) {
                    for (Map.Entry<Integer, List<double[]>> entry : knownFaceEncodings.entrySet()) {
                        for (double[] known : entry.getValue()) {
                            double score = cosineSimilarity(encoding, known);
                            if (score > bestScore) {
                                bestScore = score;
                                bestId = entry.getKey();
                            }
                        }
                    }
                } else {
                    logger.warn("Failed to generate face encoding");
                }

                Scalar boxColor;
                String labelText;

                if (bestScore > MATCH_THRESHOLD && bestId != -1) {
                    boxColor = new Scalar(0, 255, 0); // Green
                    labelText = employeeMap.get(bestId);

                    if (!markedToday.contains(bestId) && attendancePopupActive.compareAndSet(false, true)) {
                        markedToday.add(bestId);
                        logAttendance(labelText);
                        showPopup("✅ Attendance logged: " + labelText);
                        attendancePopupActive.set(false);
                    }
                } else {
                    boxColor = new Scalar(0, 0, 255); // Red
                    labelText = "Unknown";
                    long currentTime = System.currentTimeMillis();
                    if (!hasShownUnknownPopup && currentTime - lastUnknownPopupTime > UNKNOWN_POPUP_COOLDOWN_MS
                            && unknownPopupActive.compareAndSet(false, true)) {
                        hasShownUnknownPopup = true;
                        lastUnknownPopupTime = currentTime;
                        showPopup("⚠️ Unknown person detected");
                        unknownPopupActive.set(false);
                    }
                }

                Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), boxColor, 2);
                Imgproc.putText(frame, labelText, new Point(x1, y1 - 5),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, boxColor, 2);

                faceROI.release();
            }
        }

        inputBlob.release();
        detections.release();
    }

    private double[] generateFaceEncoding(Mat face) {
        if (face.empty()) {
            logger.warn("Face ROI is empty in generateFaceEncoding");
            return null;
        }
        Mat resized = new Mat();
        try {
            Imgproc.resize(face, resized, new Size(112, 112));
            Mat blob = Dnn.blobFromImage(resized, 1.0 / 255.0, new Size(112, 112), new Scalar(0, 0, 0), true, false);
            recognitionNet.setInput(blob);
            Mat features = recognitionNet.forward();
            float[] data = new float[128];
            features.get(0, 0, data);
            double[] encoding = new double[128];
            double norm = 0;
            for (float val : data) norm += val * val;
            norm = Math.sqrt(norm + 1e-8);
            for (int i = 0; i < 128; i++) encoding[i] = data[i] / norm;
            return encoding;
        } catch (Exception e) {
            logger.error("Error generating face encoding: " + e.getMessage(), e);
            return null;
        } finally {
            resized.release();
        }
    }

    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
    }

    private void saveEncodingToFile(String path, double[] enc) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            for (double d : enc) writer.write(d + "\n");
            logger.debug("Saved encoding to file: " + path);
        } catch (IOException e) {
            logger.error("Error saving encoding to file: " + path + ", " + e.getMessage());
        }
    }

    private double[] loadEncodingFromFile(String path) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(path));
            if (lines.size() < 128) {
                logger.warn("Incomplete encoding file: " + path);
                return null;
            }
            double[] enc = new double[128];
            for (int i = 0; i < 128; i++) enc[i] = Double.parseDouble(lines.get(i));
            logger.debug("Loaded encoding from file: " + path);
            return enc;
        } catch (IOException | NumberFormatException e) {
            logger.error("Error loading encoding from file: " + path + ", " + e.getMessage());
            return null;
        }
    }

    private ImageIcon matToImage(Mat mat) {
        if (mat == null) {
            logger.error("Null image passed to matToImage()");
            return null;
        }
        if (mat.empty()) {
            logger.error("Empty image passed to matToImage(), size: " + mat.size());
            return null;
        }

        MatOfByte buf = new MatOfByte();
        try {
            logger.debug("Attempting to encode Mat to JPEG, size: " + mat.size() + ", channels: " + mat.channels());
            boolean success = Imgcodecs.imencode(".jpg", mat, buf);
            if (!success) {
                logger.error("Failed to encode Mat to JPEG, size: " + mat.size());
                return null;
            }
            byte[] byteArray = buf.toArray();
            logger.debug("Encoded to JPEG, byte array length: " + byteArray.length);
            ImageIcon icon = new ImageIcon(ImageIO.read(new ByteArrayInputStream(byteArray)));
            if (icon.getIconWidth() <= 0 || icon.getIconHeight() <= 0) {
                logger.error("Invalid ImageIcon created: width=" + icon.getIconWidth() + ", height=" + icon.getIconHeight());
                return null;
            }
            logger.debug("ImageIcon created successfully: width=" + icon.getIconWidth() + ", height=" + icon.getIconHeight());
            return icon;
        } catch (IOException e) {
            logger.error("IOException in matToImage: " + e.getMessage(), e);
            return null;
        } finally {
            if (buf != null) {
                buf.release();
                logger.debug("Released MatOfByte buffer");
            }
        }
    }

    private void logAttendance(String name) {
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        String entry = name + "," + timestamp + System.lineSeparator();
        executor.submit(() -> {
            try {
                Files.writeString(Paths.get(ATTENDANCE_FILE), entry, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                logger.info("Attendance logged: " + name + " at " + timestamp);
            } catch (IOException e) {
                logger.error("Error logging attendance: " + e.getMessage(), e);
            }
        });
    }

    private void showPopup(String message) {
        SwingUtilities.invokeLater(() -> {
            JOptionPane.showMessageDialog(this, message);
            logger.info("Displayed popup: " + message);
        });
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new AttendanceSystem();
            logger.info("AttendanceSystem application started");
        });
    }
}