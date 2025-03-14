# PyCTM - Python Cognitive Toolkit for Machine Learning

## !Version

## !License

## 🌟 Overview

PyCTM implements the Cognitive Generative Planner (CGP), a hybrid planning system that combines symbolic and subsymbolic representations through a novel abstraction called "Ideas" and their vector embeddings. The CGP integrates classical cognitive architectures with deep learning approaches, specifically leveraging Transformer architectures to manipulate and generate complex plans.

The system is designed to overcome limitations in traditional planning approaches, offering enhanced creativity, flexibility, and generalization capabilities for autonomous agents operating in dynamic and complex environments.

## 🏗️ Architecture

The CGP architecture consists of several key components:

1. **Idea Representation System**: A symbolic representation framework that models knowledge as hierarchical "Idea" structures
2. **Vector Idea Serialization**: Conversion of symbolic Ideas into vector embeddings for neural processing
3. **Transformer-based Plan Generator**: A deep learning model that generates plans based on goal specifications
4. **Situated Beam Search Algorithm (SBS)**: An advanced search algorithm that efficiently explores complex planning spaces
5. **Plan Validation & Execution**: Components to validate and execute the generated plans in simulated environments

## 💡 Key Features

- Hybrid symbolic-subsymbolic architecture
- Efficient vector representation of knowledge
- Creative and diverse plan generation
- Adaptability to dynamic environments
- Low computational cost compared to traditional approaches
- Incremental learning capabilities

## 🔄 Idea Serialization and Deserialization

The core of PyCTM's hybrid approach lies in its ability to transform between symbolic and vector representations, primarily implemented through the **VectorIdeaSerializer** and **VectorIdeaDeserializer** classes.

### VectorIdeaSerializer

The `VectorIdeaSerializer` transforms hierarchical Idea objects into flat vector representations that can be processed by neural networks:

Key features of the serializer:

- **Vocabulary Management**: Dynamically builds and maintains a dictionary that maps symbolic values to numerical tokens
- **Hierarchical Encoding**: Preserves the parent-child relationships between ideas in the flattened representation
- **Type Preservation**: Encodes metadata about the data types to ensure proper reconstruction
- **Numerical Value Encoding**: Uses a specialized encoding for numerical values with high precision, maintaining both significand and exponent
- **Start/End Markers**: Adds markers to indicate the beginning and end of the serialized vector

### VectorIdeaDeserializer

The `VectorIdeaDeserializer` performs the inverse operation, reconstructing the hierarchical Idea structure from its vector representation:

Key features of the deserializer:

- **Structure Reconstruction**: Rebuilds the hierarchical structure of ideas
- **Type Recovery**: Uses encoded metadata to correctly interpret different data types
- **Value Conversion**: Accurately reconstructs numerical values from their encoded form
- **Relationship Restoration**: Reestablishes parent-child relationships between ideas

The serialization/deserialization process enables the system to:

- Feed symbolic knowledge into neural networks
- Interpret neural network outputs as structured symbolic plans
- Maintain semantic integrity across these transformations

## 🔍 Situated Beam Search Algorithm (SBS)

The SBS algorithm extends traditional beam search to incorporate contextual information from the environment, enabling more efficient exploration of the planning space. It:

- Considers occupied nodes and physical constraints
- Adapts path planning to available resources
- Evaluates plan feasibility in real-time
- Generates diverse alternative plans

## 📋 Applications

The CGP has been evaluated in simulated environments representing challenging and dynamic scenarios, demonstrating:

- High accuracy in plan generation
- Significant diversity in solution approaches
- Adaptability to changing environments
- Efficient resource utilization

## 🛠️ Installation

## 🚀 Usage

Basic usage example:

## 📖 Research Context

This project is based on the doctoral thesis titled **"Cognitive Generative Planner: A Hybrid Planning Approach Combining Symbolic and Subsymbolic Representations"**. The research explores how combining traditional AI approaches with modern deep learning techniques can create more flexible, creative and generalizable planning systems.

The CGP addresses fundamental aspects of cognitive agents including motivation and incremental learning, while overcoming limitations such as dependency on large volumes of data and high computational costs.

## 📚 Key Concepts

- **Ideas**: The fundamental knowledge representation unit in the system
- **Vector Ideas**: Numerical embeddings of Ideas that can be processed by neural networks
- **Cognitive Plan Generation**: The process of generating meaningful action sequences through a hybrid neural-symbolic approach
- **Situated Planning**: Planning that takes into account the current state of the environment and available resources

## 📄 License

This project is licensed under the MIT License.

## ✍ Author

**Eduardo de Moraes Fróes**  
Ph.D. in Computer Engineering – UNICAMP  
Advisor: Prof. Dr. Ricardo Ribeiro Gudwin
