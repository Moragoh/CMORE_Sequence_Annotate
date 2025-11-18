Aren't LLMs great?
## **0\. Rules for Labeling**
Let's start an attempt when we see a **block detached from a user's hand enter the contour zone** (let's not mark start if a block enters the contour zone but isn't detached from the hand yet, or if the block is detached but hasn't entered the contour zone yet), and stop an attempt when the block after being dropped **loses significant momentum** (so it can still be moving very little as it is at the end of it's tumble, but lets try and end the attempt earlier than later so we don't get too many frames of a block not moving at all or moving very little, which can confuse the model).

## **1\. Setup & Installation**

### **Step 1: Install Dependencies**

Open your terminal and run the following command to install all required libraries:  
```pip install ultralytics torch torchvision pandas numpy opencv-python Pillow```

### **Step 2: Run the Tool**

Open your terminal, navigate to your project folder, and run the script. You **must** specify the video and the handedness `(--R or --L)`.  
**Example for Right-Handed:**  
`python sequence_annotator.py --video my_video.mp4 --R`

#### **Command-Line Arguments:**

* **\--video**: (Required) The path to your video file.  
* **\--R or \--L**: (Required) You must choose one to set the target compartment (Right or Left).  
* **\--output**: (Optional) Specify a CSV name. By default, it saves as \[video\_name\]\_sequence\_annotations.csv.

### **Step 2: Use the Controls**

A video window will open. Use these keys to annotate:

| Key | Function | Description |
| :---- | :---- | :---- |
| **k** | **Next Frame** | Advance video by 1 frame. |
| **j** | **Prev Frame** | Rewind video by 1 frame. |
| **1** | **Mark START** | Marks the beginning of a success attempt. |
| **2** | **Mark STOP** | Marks the end of the attempt and saves the pair. |
| **q** | **Quit** | Saves all annotations to the CSV and exits. |

## **3\. Output**

The script automatically generates a CSV file (e.g., my\_video\_sequence\_annotations.csv) in your project folder with Start Frame and End Frame columns.

## **4\. Notes**
* When going backwards, if you pass a marked frame, the mark will be removed.
