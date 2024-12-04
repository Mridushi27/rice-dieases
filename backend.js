const express = require('express');
const app = express();
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });

// Define the API endpoint for image upload
app.post('/upload', upload.single('image'), (req, res) => {
    // Process the uploaded image
    const img = req.file;
    // Use the trained CNN model to predict the disease
    const prediction = predictDisease(img);
    res.json({ prediction });
});

// Define the API endpoint for disease prediction
app.post('/predict', (req, res) => {
    const img = req.body.image;
    const prediction = predictDisease(img);
    res.json({ prediction });
});

// Start the server
const port = 3000;
app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});
