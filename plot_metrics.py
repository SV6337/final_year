from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set path to your TensorBoard log file
log_dir = r"C:\Users\Lavanya\Projects\HYPERSPECTRAL_FLASK\HYPERSPECTRAL_FLASK\log\hyperspectral-0.001-2conv-basic.model"
  

# Find TensorBoard event file
log_file = None
for root, dirs, files in os.walk(log_dir):
    for file in files:
        if file.startswith("events.out.tfevents"):
            log_file = os.path.join(root, file)
            break

if not log_file:
    raise FileNotFoundError("TensorBoard event file not found!")

# Load event data
event_acc = EventAccumulator(log_file)
event_acc.Reload()

# Extract accuracy and loss data
accuracy_events = event_acc.Scalars("accuracy")
loss_events = event_acc.Scalars("loss")

# Create DataFrames
df_acc = pd.DataFrame([(e.step, e.value) for e in accuracy_events], columns=["Epoch", "Accuracy"])
df_loss = pd.DataFrame([(e.step, e.value) for e in loss_events], columns=["Epoch", "Loss"])

# Plot Accuracy vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(df_acc["Epoch"], df_acc["Accuracy"], marker='o')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("accuracy_vs_epochs.png")
plt.show()

# Plot Loss vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(df_loss["Epoch"], df_loss["Loss"], marker='o', color='red')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_vs_epochs.png")
plt.show()

