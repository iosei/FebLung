import matplotlib.pyplot as plt

# Recall values for each class
recall_values = {
    'Class 0': 1.0,
    'Class 1': 1.0,
    'Class 2': 0.998
}

# Create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(recall_values.keys(), recall_values.values(), color=['blue', 'green', 'red'])
plt.xlabel('Classes')
plt.ylabel('Recall')
plt.title('Recall for Each Class')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
