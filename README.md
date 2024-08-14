# OpenTelemetry Code Generator with GPT-2

This project fine-tunes a GPT-2 model to automatically generate OpenTelemetry instrumentation for Java code. The model is trained on a dataset of Java code snippets, and the fine-tuned model is capable of suggesting how to instrument Java methods/classes with OpenTelemetry.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Preparing the Dataset](#preparing-the-dataset)
- [Training the Model](#training-the-model)
- [Generating Instrumented Code](#generating-instrumented-code)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves fine-tuning a GPT-2 model to automatically suggest OpenTelemetry instrumentation code for Java applications. The model is trained on a dataset containing pairs of non-instrumented and instrumented Java code snippets.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gpt2-opentelemetry-codegen.git
cd gpt2-opentelemetry-codegen
```

### 2. Create a Virtual Environment

Create and activate a virtual environment using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the following packages:

```plaintext
transformers==4.30.2
datasets==2.10.1
torch==2.0.1
tensorflow==2.12.0  # Optional, if you want to use TensorFlow
```

### 4. Download the Pre-trained GPT-2 Model

The pre-trained GPT-2 model will be automatically downloaded from Hugging Face when you first run the script. No additional steps are required for this.

## Preparing the Dataset

Create a dataset file named `java_code_dataset.txt` containing pairs of non-instrumented and instrumented Java code snippets. Here is an example format:

```plaintext
#### Non-Instrumented Code Example 1:
public class ExampleService {
    public void process() {
        System.out.println("Processing data...");
    }
}

#### Instrumented Code Example 1:
// Instrumentation using OpenTelemetry
public class ExampleService {
    public void process() {
        // Start tracing
        Span span = tracer.spanBuilder("process").startSpan();
        try {
            // Original logic
            System.out.println("Processing data...");
        } finally {
            // End tracing
            span.end();
        }
    }
}
```

Ensure that the file is properly formatted and saved in the root directory of the project.

## Training the Model

### 1. Set Up Training Script

The provided script `otel_code_gen_gpt2.py` handles the fine-tuning process.

### 2. Train the Model

To fine-tune the model on your dataset, run:

```bash
python otel_code_gen_gpt2.py
```

This will:

1. Load the pre-trained GPT-2 model and tokenizer.
2. Tokenize your dataset.
3. Fine-tune the model on the dataset.
4. Save the fine-tuned model to the `./gpt2-java-instrumentation` directory.

### 3. Monitor Training

During training, you can monitor the loss and other metrics to ensure the model is learning effectively. The final model and tokenizer will be saved for future use.

## Generating Instrumented Code

After training, you can use the fine-tuned model to generate OpenTelemetry instrumentation suggestions for Java code. Modify the prompt in the script as needed:

```python
prompt = """
// Instrument this Java class with OpenTelemetry:

public class ExampleService {
    public void process() {
        System.out.println("Processing data...");
    }
}
"""

# Generate instrumented code suggestion
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=256, num_return_sequences=1)

# Decode and print the generated code
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Run the script again to see the output:

```bash
python otel_code_gen_gpt2.py
```

## Troubleshooting

- **Model Overfitting**: If the model overfits (i.e., produces very low training loss but poor results), try expanding the dataset and adjusting training parameters.
- **Memory Issues**: If you run into memory issues, reduce the batch size in the `TrainingArguments`.
- **Unexpected Behavior**: If you see warnings about the attention mask or padding tokens, ensure that the tokenizer and model are correctly configured with the `pad_token`.
