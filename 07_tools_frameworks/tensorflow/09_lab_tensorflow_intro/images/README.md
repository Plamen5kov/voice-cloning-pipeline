# Images for TensorFlow Lab

This folder should contain images used in the notebook.

## Required Image

- `onehot.png` - Illustration of one-hot encoding (width: 600px, height: 150px)

This image shows how a label vector is converted to one-hot encoding format. For example, if you have 4 classes (C=4) and labels [1, 0, 3, 2], the one-hot encoding would be:

```
Label:      1       0       3       2
One-hot:  [0]     [1]     [0]     [0]
          [1]     [0]     [0]     [0]
          [0]     [0]     [0]     [1]
          [0]     [0]     [1]     [0]
```

Where each column represents one sample, and exactly one element in each column is 1 (hot).

## How to Get the Image

### Option 1: From Course Materials
If you're taking the Coursera Deep Learning Specialization, the image is included in the assignment materials.

### Option 2: Create Your Own
You can create a simple diagram showing the one-hot encoding concept using any graphics tool or even skip this image (the notebook will still work, just won't display the illustration).

### Option 3: Download from Repository
Check if the image is available in public repositories hosting Coursera course materials (ensure you have proper licensing).

## Note

The notebook will run without this image, but you'll see a broken image link in the "Using One Hot Encodings" section. The concept is still explained in text.
