# Intelligent Signal Fault Diagnostic System

An AI-assisted fault detection and diagnostic system for **small and medium-sized machinery**.

The system takes an already-acquired machine vibration signal, analyzes it using signal processing and machine learning, identifies the likely fault condition, and provides a diagnostic explanation with recommended actions.

---

## Project Concept

In a typical machinery condition-monitoring setup, the process can be divided into two stages:

```text
┌──────────────────────────────────────┐
│        DATA ACQUISITION STAGE        │
│                                      │
│  Machine → Accelerometer → DAQ       │
│                    ↓                 │
│             MATLAB / Software         │
│                    ↓                 │
│              CSV Signal Data          │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│       THIS PROJECT STARTS HERE        │
│                                      │
│  CSV Signal                          │
│       ↓                              │
│  Signal Processing                   │
│       ↓                              │
│  Feature Extraction                  │
│       ↓                              │
│  Machine Learning Classification     │
│       ↓                              │
│  Fault Identification                │
│       ↓                              │
│  AI Diagnostic Explanation           │
│       ↓                              │
│  Recommended Action                  │
└──────────────────────────────────────┘
```

The project therefore focuses specifically on the **analysis and diagnostic stage after data acquisition**.

The accelerometer, DAQ hardware, and data-acquisition software are considered external to this project. The system assumes that the resulting vibration data is available in a suitable digital format such as CSV.

---

## What Does the System Do?

The system answers three main questions:

### 1. Is the machine operating normally?

The incoming vibration signal is analyzed to determine whether it corresponds to a normal operating condition or an abnormal condition.

### 2. If there is a fault, what type is it?

The machine-learning model currently classifies the signal into:

* **Normal**
* **High Vibration**
* **Harmonic Fault**
* **Impulsive Fault**

### 3. What should be done?

The diagnostic layer interprets the detected condition and provides a human-readable explanation of what the signal may indicate and what should be investigated or considered as the next maintenance action.

---

## System Workflow

```text
        Vibration CSV
              │
              ▼
      Signal Processing
              │
              ▼
      Feature Extraction
              │
              ▼
       KNN Classifier
              │
              ▼
      Fault Classification
              │
              ▼
    Diagnostic Interpretation
              │
              ▼
      AI-Assisted Explanation
              │
              ▼
       Recommended Action
```

The key idea is to move beyond simply detecting an abnormal signal.

> **The goal is to transform machine vibration data into understandable diagnostic information.**

---

## Machine Learning

The current classification system uses a **K-Nearest Neighbors (KNN)** model.

Rather than feeding the raw vibration waveform directly into the classifier, relevant signal features are extracted first. These features are then used by the machine-learning model to determine the most likely operating condition.

The original machine-learning workflow was initially developed and validated in **MATLAB**. It was subsequently transitioned to **Python** and integrated into the final diagnostic application.

The original MATLAB implementation is retained in the repository for reference.

---

## AI Diagnostic Layer

Machine learning provides the classification, while the AI layer helps explain the result.

```text
Machine Learning
       │
       ▼
"Impulsive Fault"
       │
       ▼
AI Diagnostic Layer
       │
       ├── What was detected?
       ├── What could it indicate?
       ├── What should be inspected?
       └── What action should be considered?
```

This creates a distinction between:

**ML → Detection and Classification**

**AI → Explanation and Diagnostic Guidance**

The AI layer therefore complements the machine-learning model rather than replacing it.

---

## Application Architecture

```text
CSV Signal
    │
    ▼
┌─────────────────────┐
│  Feature Extraction │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   KNN Classification│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Fault Interpretation │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   AI Explanation    │
└──────────┬──────────┘
           ▼
    Diagnostic Result
```

The repository contains the complete executable application, allowing the system to be used without reproducing the development environment.

---

## Project Structure

```text
signal-fault-app/
│
├── backend/
│   ├── features.py
│   ├── signal_generator.py
│   ├── train_model.py
│   ├── classifier.py
│   ├── agent.py
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── matlab_scripts/
    └── Original MATLAB implementation
```

---

## Potential Applications

The concept is intended for **small and medium-sized rotating machinery**, including applications such as:

* Motors
* Pumps
* Fans
* Bearings
* Small rotating assemblies
* Workshop machinery

The actual diagnostic capability depends on the quality and representativeness of the available training and test data.

---

## Limitations

This is a **prototype diagnostic system**. The current model supports a limited number of fault classes and is dependent on the characteristics of the data used during development.

For deployment on real machinery, the system would require validation using representative real-world sensor data and machine-specific operating conditions.

The diagnostic recommendations should be treated as **engineering guidance**, not as a replacement for physical inspection or qualified maintenance personnel.

---

## Future Development

Potential extensions include:

* Additional mechanical and electrical fault classes
* Real-time sensor integration
* Fault severity estimation
* Historical machine-condition tracking
* Multiple-sensor analysis
* Predictive maintenance
* Edge/embedded deployment
* Machine-specific model training

---

## Project Objective

The ultimate objective is to develop an accessible intelligent diagnostic layer that sits **after conventional vibration data acquisition** and converts raw machine signals into useful maintenance information.

```text
Machine
   ↓
Accelerometer
   ↓
DAQ System
   ↓
Data Acquisition Software
   ↓
CSV
   ↓
★ THIS PROJECT ★
   ↓
Fault Detection
   ↓
Fault Classification
   ↓
Diagnosis
   ↓
Recommended Action
```

**From vibration data to actionable diagnosis.**
