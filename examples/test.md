# Robot Arm Assembly Instructions

Follow these steps to assemble your robot arm.

---

## Step 1: Unbox the Components
Unpack the components and verify you have all the required parts.

![Unboxing](images/unboxing.gif)

---

### Navigation
<button onclick="location.href='#step-2-assemble-the-base';" style="padding: 10px 20px; font-size: 16px;">Next Step</button>

---

## Step 2: Assemble the Base
Attach the base to the platform using the provided screws.

![Assembling Base](images/assembling-base.gif)

---

### Navigation
<button onclick="location.href='#step-1-unbox-the-components';" style="padding: 10px 20px; font-size: 16px;">Previous Step</button>
<button onclick="location.href='#step-3-attach-the-arm-segments';" style="padding: 10px 20px; font-size: 16px;">Next Step</button>

---

## Step 3: Attach the Arm Segments
1. Connect each segment in order, ensuring they are aligned.
2. Tighten the screws securely.

![Attaching Arm Segments](images/attaching-arm.gif)

---

### Navigation
<button onclick="location.href='#step-2-assemble-the-base';" style="padding: 10px 20px; font-size: 16px;">Previous Step</button>
<button onclick="location.href='#step-4-initialize-the-motor-controller';" style="padding: 10px 20px; font-size: 16px;">Next Step</button>

---

## Step 4: Initialize the Motor Controller
Install the motor driver software and upload the configuration script:

```python
import tinymovr

controller = tinymovr.Controller('can0')
controller.calibrate_motor()
