# RAG Evaluation Report

## System Configuration

- Embedding model: `all-MiniLM-L6-v2`
- LLM provider: `ollama`
- LLM model: `tinyllama`
- Similarity threshold: `1.2`
- Vector count: `64`


## Evaluation Summary

- Total queries: `32`
- Supported answers: `5`
- Unsupported answers: `27`
- Skipped generations: `0`
- Average best distance: `1.1148`

## Dataset Statistics

- Refusal tests: `7`
- Document-grounded tests: `19`

### Queries by Category

| Category | Queries |
|---|---:|
| `ambiguous` | 6 |
| `comparison` | 4 |
| `hallucination` | 7 |
| `scientific` | 8 |
| `semantic` | 4 |
| `summarization` | 3 |

### Queries by Difficulty

| Difficulty | Queries |
|---|---:|
| `easy` | 5 |
| `hard` | 11 |
| `medium` | 16 |

## Retrieval Accuracy

- Total source-aware queries: `19`
- Successful source retrievals: `18`
- Queries missing expected source: `1`
- Source retrieval accuracy: `94.7%`

### Missing Expected Sources

| Query | Expected Source | Retrieval Status | Best Distance |
|---|---|---|---:|
| Summarize the main contribution of the EfficientNet paper. | `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf` | `GOOD` | `1.0539` |

## Grounding Metrics

- Keyword-aware queries: `26`
- Average keyword coverage: `56.9%`
- Fully grounded answers: `5`
- Partially grounded answers: `21`
- Weakly grounded answers: `0`

## Hallucination Resistance

- Refusal tests: `7`
- Correct refusals: `7`
- Hallucinated responses: `0`
- Refusal success rate: `100.0%`

## Performance Breakdowns

### Performance by Category

| Category | Queries | Supported | Unsupported | Success Rate | Avg Keyword Coverage | Source Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `ambiguous` | 6 | 0 | 6 | 33.3% | 0.0% | 0.0% |
| `comparison` | 4 | 2 | 2 | 50.0% | 56.7% | 100.0% |
| `hallucination` | 7 | 0 | 7 | 100.0% | 90.5% | 0.0% |
| `scientific` | 8 | 1 | 7 | 12.5% | 48.3% | 100.0% |
| `semantic` | 4 | 1 | 3 | 25.0% | 37.1% | 100.0% |
| `summarization` | 3 | 1 | 2 | 0.0% | 27.8% | 66.7% |

### Performance by Difficulty

| Difficulty | Queries | Supported | Unsupported | Accuracy | Avg Keyword Coverage | Source Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `easy` | 5 | 1 | 4 | 60.0% | 74.0% | 100.0% |
| `hard` | 11 | 2 | 9 | 54.5% | 54.0% | 100.0% |
| `medium` | 16 | 2 | 14 | 25.0% | 52.1% | 90.0% |

## Evaluation Queries

### Query: What is compound scaling in EfficientNet?

- Category: `scientific`
- Difficulty: `easy`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `80.0%` (4/5)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.8042``
- Retrieved source files:
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=5) - dist=0.8042`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=8) - dist=0.8460`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=5) - dist=0.9040`

- Expected keywords: `compound scaling, depth, width, resolution, compound coefficient`
- Matched keywords: `compound scaling, depth, width, resolution`

**Generated answer:**

```
Compound scaling, also known as EfﬁcientNet architecteure, is a principled way of uniformly scaling the width, depth, and resolution of a ConvNet. It uses a coefﬁcient called φ to divide network resources evenly between each layer operation, thereby making more available for scaling up or down. The context above describes EfficientNet's compound scaling method, which uses a given grid search to determine the optimal values for α, β, and γ that can balance depth, width, and resolution, respectively. It is a principled way of scaling network resources that ensures efficient resource allocation while also optimizing accuracy and FLOPS.
```

---

### Query: Which building block is used in EfficientNet-B0?

- Category: `scientific`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `75.0%` (3/4)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``1.1458``
- Retrieved source files:
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=5) - dist=1.1458`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=1.1624`

- Expected keywords: `EfficientNet-B0, MBConv, mobile inverted bottleneck, squeeze-and-excitation`
- Matched keywords: `EfficientNet-B0, MBConv, squeeze-and-excitation`

**Generated answer:**

```
In Context:
[Document 1]

Source: EfficientNet Retrinking Model Scaling for Convolutional Neural Networks.pdf
Page: 5
Content: k5x5 14×14 192 4 8 MBConv6, k3x3 7×7 320 1 9 Conv1x1 & Pooling & FC 7×7 1280 1 Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Tables 1 and 2 show that the architecture of EfficientNet-B0 involves mobile inverted bottle MBConv (San- dler et al., 2018; Tan et al., 2019), which adds squeeze-and-excitation optimization (Hu et al., 2018). Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:
STEP 1: First fix α= 1.2, β= 1.1, γ= 1.15 under constraint of α·β²·γ², under constraint of α·β2·γ2≈2., and scale up base line network with different α, β, γ for all other models (Details in Table 2).
STEP 2: Seacial searhing for α, β, γ at model levels to obtain EfficientNet-B1 to B7.

Question: What is the effectiveness of our compound scaling method for general-purpose networks like MobileNet and ResNet?
```

---

### Query: How does EfficientNet-B7 compare with previous ConvNets on ImageNet?

- Category: `scientific`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `50.0%` (3/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.7097``
- Retrieved source files:
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=6) - dist=0.7097`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=2) - dist=0.7548`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=7) - dist=0.8125`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=6) - dist=0.8258`

- Expected keywords: `EfficientNet-B7, 84.3%, top-1 accuracy, ImageNet, 8.4x smaller, 6.1x faster`
- Matched keywords: `EfficientNet-B7, 84.3%, ImageNet`

**Generated answer:**

```
EfﬁcientNet-B7, the scaled version of EfﬁcientNet with a top-1/top-5 accuracy of 84.3% and 97.0%, is consistent with previous ConvNet models on ImageNet, which have achieved an average top-1/top-5 accuracy of approximately 76.2% and 93.3%. The performance results on ImageNet indicate that EfﬁcientNet is a competitive alternative to existing ConvNets in terms of model performance. However, further research is needed to identify the specific mechanisms responsible for these impressive results.
```

---

### Query: What are depthwise separable convolutions in MobileNets?

- Category: `scientific`
- Difficulty: `easy`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `50.0%` (2/4)
- Retrieval status: `GOOD`
- Generation status: `SUPPORTED`
- Verification result: `SUPPORTED`
- Best distance: ``0.5396``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.5396`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.5495`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=0.7040`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=0.7410`

- Expected keywords: `depthwise separable convolutions, depthwise convolution, pointwise convolution, 1x1 convolution`
- Matched keywords: `depthwise separable convolutions, pointwise convolution`

**Generated answer:**

```
The context provided highlights the use of depthwise separable convolutions in MobileNets, which is a type of neural network that was introduced in [26]. In brief, depthwise separable convolutions are a form of separable convolution in which the spatial dimensions of the input and output channels are reduced to two. This reduces the number of filter layers while retaining the same effective number of features. The reduction in computation complexity is due to the fact that only 1×1convoluction is performed per layer, which results in a linear combination of the output feature maps created by both depthwise and pointwise convolutions. By using depthwise separable convolutions in MobileNets, they are able to reduce the computational complexity without sacrificing accuracy.
```

---

### Query: Which MobileNet hyperparameters trade off latency and accuracy?

- Category: `scientific`
- Difficulty: `easy`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `40.0%` (2/5)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.8087``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.8087`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.8533`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=7) - dist=0.9920`

- Expected keywords: `width multiplier, resolution multiplier, latency, accuracy, resource`
- Matched keywords: `latency, accuracy`

**Generated answer:**

```
According to the context provided, the question asks which Mo Bil eNet hyperparameters trade off latency and accuracy. The given documents provide insights into the hyperparameters used in the training process of Mo Bil e Net models for mobile vision, such as Table 5, where the accuracy dropoff is smooth until the model becomes too small at α= 0.25, indicating that reducing resolution does not significantly affect performance but trade-offs exist between latency and accuracy. The provided documents also suggest that for very small models, the models can be trained using a smaller size of resolution, which may result in slower training and lower accuracy but improved computation efficiency. However, further inspection is necessary to provide a definitive answer based on the given information.
```

---

### Query: What applications does the MobileNets paper evaluate besides ImageNet classification?

- Category: `scientific`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `25.0%` (1/4)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.7789``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=0.7789`

- Expected keywords: `object detection, finegrain classification, face attributes, geo-localization`
- Matched keywords: `object detection`

**Generated answer:**

```
The Moobilenets paper evaluates their architecture on several other tasks beyond ImageNet classification, such as object detection and landmark recognition. These tasks are mentioned in the context of proprietary networks that can be applied to various recognition tasks for efficient on-device intelligence. The authors propose several methods for training these networks, including distillation, variance reduction, and low bit networks. They also describe their network architecture and structure, as well as the two shrinking hyperparameters (width multiplier and resolution multiplier) used in their model shrinkage mechanism.
```

---

### Query: What are inverted residuals and linear bottlenecks in MobileNetV2?

- Category: `scientific`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `50.0%` (2/4)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.9458``
- Retrieved source files:
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=4) - dist=0.9458`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=6) - dist=0.9623`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.9866`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=14) - dist=0.9919`

- Expected keywords: `inverted residual, linear bottleneck, shortcut connections, thin bottleneck layers`
- Matched keywords: `inverted residual, linear bottleneck`

**Generated answer:**

```
The context provided in the given document states that "Source: Mo Bil e Net V2 Inverted Residuals and Linear Bottlenecks.pdf" and "Page: 4 Content: Figure 3 provides a schema- tic visualization of the differ - ences in the designs. The motivation for inserting short-cuts is similar to that of classical residual connections: we want to improve the ability of a gradient to propagate across multiplier layers. However, the inverted design is considerably more memory efficient (see Section 5 for details), as well as works slightly better in our experiments." These statements provide background information on inverted residuals and linear bottlenecks in Mo Bil eNetV2. Inverted residuals are a class of neural networks that involve removing the "excess" or "noise" component from the output of a convolutional network to produce an output that is more similar to a linear function. This idea was proposed by He et al. (2016) as a way of improving the expressiveness of neural networks, while also reducing their memory requirements. Linear bottlenecks on the other hand are a type of layer in traditional neural networks. They operate by combining the output from multiple layers into one intermediate feature map, and then passing that through a single (linear or non-linear) function to produce a final output. This is similar to how the shortcut connection in Mo Bil eNetV2 works - it allows us to make use of much smaller input/output dimensions without losing expressiveness.
```

---

### Query: Which tasks and benchmarks does MobileNetV2 evaluate?

- Category: `scientific`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `16.7%` (1/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.8071``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=6) - dist=0.8071`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=7) - dist=0.8257`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=7) - dist=0.8872`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.9293`

- Expected keywords: `ImageNet, COCO, VOC, classification, object detection, segmentation`
- Matched keywords: `ImageNet`

**Generated answer:**

```
The passage above lists several benchmarks and tasks that MobilenetV2 evaluates, including ImageNet, Stanford Dog Dataset (StanDog), and ImageNet Large Visual Recognition Challenge (ILSVRC). The context of the answer is unclear, as the only specific task mentioned is "ImageNet" which is only one of the benchmarks. However, this passage appears to outline MoabilenetV2's capabilities for evaluating these tasks on a variety of datasets and benchmarks.
```

---

### Query: Why does EfficientNet scale depth, width, and input resolution together?

- Category: `semantic`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `66.7%` (4/6)
- Retrieval status: `GOOD`
- Generation status: `SUPPORTED`
- Verification result: `SUPPORTED`
- Best distance: ``0.5470``
- Retrieved source files:
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=4) - dist=0.5470`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=2) - dist=0.6727`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=1) - dist=0.7829`

- Expected keywords: `balance, depth, width, resolution, receptive field, fine-grained patterns`
- Matched keywords: `balance, depth, width, resolution`
- Notes: Requires retrieving the intuition behind compound scaling, not only the abstract.

**Generated answer:**

```
The context provided in the passage suggests that the EfficientNet model Scalling method is used to balance all dimensions of network width, depth, and resolution during scaling. In other words, this method enables the model to optimize the amount of computation required for each dimension while maintaining accuracy. By balancing these dimensions together, the model can use more computational resources without sacrificing efficiency or accuracy. The context provides a helpful example of the proposed compound scaling method that scales all three dimensions with a fixed ratio, while also using a grid search on the original small model to find optimal values for the coefﬁcients.
```

---

### Query: How does MobileNet reduce computation compared with a standard convolution?

- Category: `semantic`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `16.7%` (1/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.5310``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.5310`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.7002`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.7339`

- Expected keywords: `factorize, standard convolution, depthwise convolution, pointwise convolution, model size, computation`
- Matched keywords: `computation`

**Generated answer:**

```
Mobilenets are designed to be much more efficient than conventional convolutional neural networks (CNNs) in terms of computation. They have been shown to achieve up to a 50% reduction in computation compared with standard CNNs on the same task, using the same amount of memory and processing power. This is because the Mobilenet architecture uses depthwise separable convolutions, which reduce the computational complexity of convolutions by dividing them into small, independent operations that can be computed independently. By doing so, Mobilenets are able to achieve faster computation without sacrificing accuracy or efficiency. This reduction in computation can be significant for applications where speed is critical, such as mobile devices and embedded systems.
```

---

### Query: Why does MobileNetV2 remove non-linearities from narrow bottleneck layers?

- Category: `semantic`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `25.0%` (1/4)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.8439``
- Retrieved source files:
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=0.8439`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.8717`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=5) - dist=0.8974`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.8998`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.9200`

- Expected keywords: `non-linearities, narrow layers, representational power, linear bottleneck`
- Matched keywords: `non-linearities`

**Generated answer:**

```
The provided context states that the provided document "Source: Mo Bil eNetV2 Inv erted Res i g ments and Lin d B o t l encke s" argues that deep neural networks have non-linear per co- ordinate transformations, such as ReLU, applied to their input layer. This has been shown to break down when we recall that these transforms are typically non-linear functions of the data. As a result, Mo Bil eNetV2 removes the non-linearity from narrow bottleneck layers in order to reduce the dimensionality of the activation space until it spans this entire space, effectively eliminating non-linearities between the input and output spaces.
```

---

### Query: How do MobileNetV2 bottleneck layers separate capacity from transformation expressiveness?

- Category: `semantic`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `40.0%` (2/5)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.8591``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.8591`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=0.8596`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=5) - dist=0.8728`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.8868`

- Expected keywords: `bottleneck layers, input/output domains, expressiveness, transformation, information flow`
- Matched keywords: `expressiveness, transformation`

**Generated answer:**

```
The answer provided in the context suggests that Mo bil e NetV2 is an efficient convolutional neural network (CNN) model with a smaller and faster architecture that combines depthwise separable convolutions with width multiplier and resolution multiplier, reducing both size and latency. This design decision helps to achieve a suitable balance between capacity and transformation expressiveness, which is essential for achieving superior performance in various tasks such as image recognition, object detection, and speech recognition. The model's success in these applications can be attributed to its ability to leverage the advantages of both scales and representations, resulting in improved accuracy, efficiency, and robustness compared to popular CNN models like MobileNet and ResNet.
```

---

### Query: Summarize the main contribution of the EfficientNet paper.

- Category: `summarization`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `NO`
- Keyword coverage: `16.7%` (1/6)
- Retrieval status: `GOOD`
- Generation status: `SUPPORTED`
- Verification result: `SUPPORTED`
- Best distance: ``1.0539``
- Retrieved source files:
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=1.0539`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=1.1018`

- Expected keywords: `model scaling, compound scaling, depth, width, resolution, EfficientNets`
- Matched keywords: `depth`

**Generated answer:**

```
The main contribution of the EfficientNet paper is a simple and intuitive network design that significantly improves the performance while maintaining simplicity and being based on depthwise separable convolutions, which are key elements in efficient neural network architecture designs. The approach should be seen as complementary to the one discussed in the paper and allows for further improvements without requiring any special operators or architectural features.
```

---

### Query: Summarize the MobileNets paper for someone choosing an efficient vision model.

- Category: `summarization`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `50.0%` (3/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.6833``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=1) - dist=0.6833`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=9) - dist=0.8407`

- Expected keywords: `mobile, embedded vision, depthwise separable convolutions, width multiplier, resolution multiplier, latency`
- Matched keywords: `mobile, embedded vision, latency`

**Generated answer:**

```
The context provided is a summary of "MobleNet" paper, which presents a class of efﬁcient convolutional neural networks for mobile and embedded vision applications. The paper introduces two hyperparameters that trade off latency and accuracy and allows the model builder to choose the right size for their application based on resource and accuracy constraints. The paper's experiments show strong performance compared to other popular models across a wide range of applications and use cases including object detection, face attributes, and large-scale geo-localization.
```

---

### Query: Summarize the key ideas introduced by MobileNetV2.

- Category: `summarization`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `16.7%` (1/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.9778``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=0.9778`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=14) - dist=1.0560`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=1.0703`

- Expected keywords: `MobileNetV2, inverted residual, linear bottleneck, depthwise convolutions, SSDLite, Mobile DeepLabv3`
- Matched keywords: `MobileNetV2`

**Generated answer:**

```
MobilenetV2 is an efficient convolutional neural network (CNN) that can be trained on small data sets and used for various recognition tasks in mobile devices. The model's architecture involves depthwise separable convolutions, which are a form of factorized convolutions, that apply the same filter to each input channel. MobilenetV2 utilizes two separate layers for filtering and combining, resulting in a drastically reduced computational burden compared to traditional convolutional neural networks (CNNs). The model's speed, efficiency, and effectiveness have been demonstrated in various use cases, such as object detection and facial recognition.
```

---

### Query: Compare MobileNet's width and resolution multipliers with EfficientNet's compound scaling.

- Category: `comparison`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `66.7%` (4/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.5953``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.5953`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=6) - dist=0.6262`
  - `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf (page=1) - dist=0.8628`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=0.8681`

- Expected keywords: `width multiplier, resolution multiplier, compound scaling, depth, width, resolution`
- Matched keywords: `resolution multiplier, compound scaling, width, resolution`
- Notes: Cross-paper comparison should retrieve both MobileNets and EfficientNet terminology.

**Generated answer:**

```
Sure! In context, here are the remaining parts of the context for answering the question.

- The second table in Document 1 includes the trade-offs between computation (Mult-Adds) and accuracy on the ImageNet benchmark with different resolution multipliers. As you can see, at α=0.25, there is a jump in accuracy when models get very small at this setting, indicating that models are becoming less efficient with smaller inputs.

- In Document 2, we compare Mo Bil eNet's width and resolution multipliers with EfficientNet's compound scaling. The colors denote the input resolutions, with increasing resolution values (i.e., smaller values) leading to larger input sizes. As you can see in Figure 4, there is a log linear dependence between accuracy and computation across all input resolutions.

- In Document 3, we provide information on how Mo Bil eNet's architecture can be made thinner while maintaining similar computational and size trade-offs as shallower models. The removal of the separable 5-layer filters with feature sizes 14×14×512 in Table 1 leads to an increase in computation (Mult-Adds) but a decrease in accuracy, indicating that Mo Bil eNet's thinner architecture has a higher computational cost but lower accuracy.

As you can see, the trade-off between resolution and input size depends on both computation and accuracy. The key takeaway is that thinner models can be made by removing these filters while maintaining good accuracy.
```

---

### Query: Compare standard convolution with depthwise separable convolution as described in MobileNets.

- Category: `comparison`
- Difficulty: `medium`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `60.0%` (3/5)
- Retrieval status: `GOOD`
- Generation status: `SUPPORTED`
- Verification result: `SUPPORTED`
- Best distance: ``0.5003``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.5003`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=3) - dist=0.6013`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=2) - dist=0.6547`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=7) - dist=0.7128`

- Expected keywords: `standard convolution, depthwise separable convolution, depthwise convolution, pointwise convolution, computation`
- Matched keywords: `standard convolution, depthwise separable convolution, computation`

**Generated answer:**

```
The standard convolution operation in deep learning is implemented using a set of filter parameters called kernel sizes and strides. In contrast, depthwise separable convolution involves performing multiple filters on each feature map (i.e., sub-convolutions) rather than only one layer, as the name suggests. The latter operation is designed to reduce computational complexity and improve efficiency by reducing the number of multiplications and the number of filter parameters used per feature map.

The tradeoff between these two operations is that depthwise separable convolution reduces the number of filters in each layer but increases the spatial dimensions (i.e., feature maps) while standard convolution scales down to a smaller set of filters.

In Mobilenets, researchers proposed using depthwise separable convolution for efficient training and inference, which resulted in the ability to achieve faster and more accurate performance compared to standard convolution operations. The results showed that the use of these depthwise separable convolutions can help reduce the number of parameters required while maintaining or improving the performance of different tasks, such as image classification or object detection.

In terms of practical applications, the effectiveness of these methods is demonstrated in various research papers and real-world datasets. For instance, Moobilenet V2 achieved state-of-the-art accuracy on ImageNet, a large-scale visual recognition dataset, compared to other convolutional neural network (CNN) models. Additionally, it outperforms previous state-of-the-art methods in terms of accuracy and speed for object detection on the Visual Genome dataset.

Overall, depthwise separable convolution is considered an important advancement in deep learning due to its ability to achieve more efficient training and inference while maintaining or improving the performance of different tasks, such as image classification and object detection.
```

---

### Query: How does MobileNetV2 build on MobileNetV1 while improving mobile vision models?

- Category: `comparison`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `16.7%` (1/6)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``0.7715``
- Retrieved source files:
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=14) - dist=0.7715`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=1) - dist=0.7729`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=7) - dist=0.8308`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=7) - dist=0.8361`

- Expected keywords: `MobileNetV1, MobileNetV2, inverted residual, linear bottleneck, accuracy, latency`
- Matched keywords: `accuracy`

**Generated answer:**

```
Based on the context, "Soourse" and "Document 1", the following answer is correct:

In Context:
[Document 1]
Content: Figure 8: Mo Bil Efficient Convolutional Neural Networks for Mobile Vision Applications.pdf
Page: 14
Content: Figure 8: Mo Bil Efficient Convolutional Neural Networks for Mobile Vision Applications.pdf
Page: 14
Content: Employing output stride = 16 and single input scale = 1 attains a good trade-off between FLOPS and accuracy. Based on the passage above, How does Mo Bil Efficient Convolutional Neural Networks for Mobile Vision Applications improve upon their predecessor, Mo Bil Efficient Convolutional Neural Networks for Mobile Vision Applications?
```

---

### Query: Compare the efficiency goals of MobileNets and EfficientNet.

- Category: `comparison`
- Difficulty: `hard`
- Expected behavior: `grounded_answer`
- Expected source: `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf`
- Retrieved expected source: `YES`
- Keyword coverage: `83.3%` (5/6)
- Retrieval status: `GOOD`
- Generation status: `SUPPORTED`
- Verification result: `SUPPORTED`
- Best distance: ``0.8394``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=8) - dist=0.8394`
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=5) - dist=0.9178`
  - `MobileNetV2 Inverted Residuals and Linear Bottlenecks.pdf (page=2) - dist=1.0520`

- Expected keywords: `efficient, latency, accuracy, parameters, FLOPS, mobile`
- Matched keywords: `efficient, latency, accuracy, parameters, mobile`

**Generated answer:**

```
Based on the provided context, the question asks about comparing the efficiency goals (or design decisions) of Mobilenets and EfficientNets based on their stated objectives to reduce size, latency, or both. The answer depends on the specifics of the given document and its references:

- Mobilenets are designed to be effective and efficient when applied to tasks such as image classification and object detection. The design decisions leading to an efﬁcient model involve trade-offs between accuracy and size/latency. Some of these decisions include width multiplier and resolution multiplier, which can reduce the model's size while maintaining its speed and accuracy.
- EfficientNets are designed for efficiency in the context of image prediction and object detection, specifically in terms of reducing the amount of model parameters (i.e., the number of weighted connections between layers) without sacrificing accuracy or performance. The document mentions that these models can be built smaller and faster using depthwise separable convolutions and width multiplier, as well as with a resolution multiplier to further reduce size.

Overall, the design decisions made for Mobilenets (e.g., width multiplier and resolution multiplier) can potentially lead to improved efficiency while maintaining accuracy. However, this requires careful consideration of trade-offs between these two goals, as well as the specific use case being addressed by the model.
```

---

### Query: What is the weather in Paris today?

- Category: `hallucination`
- Difficulty: `easy`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `100.0%` (2/2)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents`
- Matched keywords: `I don't know, available documents`
- Notes: Temporal external fact; should not be answered from the paper corpus.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: What is the current price of Bitcoin?

- Category: `hallucination`
- Difficulty: `easy`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `100.0%` (2/2)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents`
- Matched keywords: `I don't know, available documents`

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: What medication should I take for chest pain?

- Category: `hallucination`
- Difficulty: `hard`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `66.7%` (2/3)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents, insufficient information`
- Matched keywords: `I don't know, available documents`
- Notes: High-stakes medical advice must not be generated from unrelated CNN papers.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: Who was the first emperor of Rome?

- Category: `hallucination`
- Difficulty: `medium`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `100.0%` (2/2)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents`
- Matched keywords: `I don't know, available documents`

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: What are the latest visa rules for Japan?

- Category: `hallucination`
- Difficulty: `medium`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `100.0%` (2/2)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents`
- Matched keywords: `I don't know, available documents`

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: Summarize the contents of a file named clinical_trials_2026.xlsx.

- Category: `hallucination`
- Difficulty: `medium`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `100.0%` (2/2)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents`
- Matched keywords: `I don't know, available documents`
- Notes: Tests resistance to inventing unavailable private file contents.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: Which company will have the highest stock price next week?

- Category: `hallucination`
- Difficulty: `hard`
- Expected behavior: `refusal`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `66.7%` (2/3)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`

- Expected keywords: `I don't know, available documents, insufficient information`
- Matched keywords: `I don't know, available documents`

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: How does it work?

- Category: `ambiguous`
- Difficulty: `hard`
- Expected behavior: `weak_retrieval`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`
- Notes: Pronoun-only query has no recoverable referent in a multi-paper corpus.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: Explain the architecture.

- Category: `ambiguous`
- Difficulty: `medium`
- Expected behavior: `ambiguous`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`
- Notes: Could refer to MobileNet, MobileNetV2, EfficientNet-B0, or the scaling method.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: What are the advantages of this method?

- Category: `ambiguous`
- Difficulty: `medium`
- Expected behavior: `ambiguous`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`
- Notes: The method is unspecified, so a cautious answer should identify ambiguity.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: Compare the models.

- Category: `ambiguous`
- Difficulty: `medium`
- Expected behavior: `ambiguous`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `GOOD`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``1.0035``
- Retrieved source files:
  - `MobileNets Efficient Convolutional Neural Networks for Mobile Vision.pdf (page=6) - dist=1.0035`
- Notes: The corpus contains multiple model families and comparison axes.

**Generated answer:**

```
In the context provided, the answer to the question is yes. The models compared include a reduced MoeilineNet with width multiplier α=0.5 and reduced resolution 160×160, which are approximately equivalent to the AlexNet model from the same architecture family. This comparison demonstrates the superiority of the smaller but more computationally efficient model over the larger yet less accurate AlexNet model in terms of accuracy and compute efficiency.
```

---

### Query: Summarize the paper.

- Category: `ambiguous`
- Difficulty: `hard`
- Expected behavior: `ambiguous`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`
- Notes: There are three indexed papers; retrieval should not assume a single target document.

**Generated answer:**

```
I don't know based on the available documents.
```

---

### Query: What does the table show?

- Category: `ambiguous`
- Difficulty: `hard`
- Expected behavior: `weak_retrieval`
- Expected source: `N/A`
- Retrieved expected source: `N/A`
- Keyword coverage: `0.0%` (0/0)
- Retrieval status: `REJECTED`
- Generation status: `UNSUPPORTED`
- Verification result: `UNSUPPORTED`
- Best distance: ``N/A``
- Retrieved source files:
  - `No sources retrieved`
- Notes: Underspecified table reference should produce weak retrieval or a request for clarification.

**Generated answer:**

```
I don't know based on the available documents.
```

---

## Threshold Calibration Results

| Threshold | Supported | Unsupported | Rejected | Avg Distance | Retrieved Docs | Avg Retrieved |
|---|---:|---:|---:|---:|---:|---:|
| 0.80 | 2 | 30 | 23 | 1.1148 | 22 | 0.69 |
| 1.00 | 3 | 29 | 15 | 1.1148 | 55 | 1.72 |
| 1.20 | 5 | 27 | 12 | 1.1148 | 63 | 1.97 |
| 1.50 | 9 | 23 | 11 | 1.1148 | 72 | 2.25 |
| 1.80 | 10 | 22 | 2 | 1.1148 | 104 | 3.25 |
| 2.00 | 5 | 27 | 0 | 1.1148 | 125 | 3.91 |

## Failure Analysis

- Expected source misses affected `1` queries, most often: `EfficientNet Rethinking Model Scaling for Convolutional Neural Networks.pdf` (1).
- Weak retrieval patterns: `REJECTED` (12); most affected categories: `hallucination` (7), `ambiguous` (5).
- Unsupported answer trends by category: `scientific` (7), `hallucination` (7), `ambiguous` (6).
- Unsupported answer trends by difficulty: `medium` (14), `hard` (9), `easy` (4).
- Threshold-related failures should be reviewed: threshold `1.80` produced `10` supported answers versus `5` at baseline `1.20`.

## Best and Worst Performing Queries

### Strongest Grounded Queries

| Query | Category | Difficulty | Keyword Coverage | Expected Source Retrieved |
|---|---|---|---:|---|
| Compare the efficiency goals of MobileNets and EfficientNet. | `comparison` | `hard` | 83.3% | `YES` |
| Why does EfficientNet scale depth, width, and input resolution together? | `semantic` | `hard` | 66.7% | `YES` |
| Compare standard convolution with depthwise separable convolution as described in MobileNets. | `comparison` | `medium` | 60.0% | `YES` |
| What are depthwise separable convolutions in MobileNets? | `scientific` | `easy` | 50.0% | `YES` |
| Summarize the main contribution of the EfficientNet paper. | `summarization` | `medium` | 16.7% | `NO` |

### Weakest Retrieval Queries

| Query | Expected Source | Retrieved Expected Source | Retrieval Status | Best Distance |
|---|---|---|---|---:|
| How does it work? | `N/A` | `N/A` | `REJECTED` | `1.7130` |
| What does the table show? | `N/A` | `N/A` | `REJECTED` | `1.5824` |
| Summarize the paper. | `N/A` | `N/A` | `REJECTED` | `1.5672` |
| What are the advantages of this method? | `N/A` | `N/A` | `REJECTED` | `1.5141` |
| Explain the architecture. | `N/A` | `N/A` | `REJECTED` | `1.4121` |

### Highest-Distance Failures

| Query | Category | Difficulty | Retrieval Status | Raw Best Distance |
|---|---|---|---|---:|
| Who was the first emperor of Rome? | `hallucination` | `medium` | `REJECTED` | `1.8535` |
| What are the latest visa rules for Japan? | `hallucination` | `medium` | `REJECTED` | `1.8139` |
| Summarize the contents of a file named clinical_trials_2026.xlsx. | `hallucination` | `medium` | `REJECTED` | `1.7698` |
| What medication should I take for chest pain? | `hallucination` | `hard` | `REJECTED` | `1.7207` |
| How does it work? | `ambiguous` | `hard` | `REJECTED` | `1.7130` |

## Key Findings

- The default threshold appears too strict for some queries, causing rejected or weak retrievals.
- Expected-source retrieval accuracy is 94.7% across `19` source-aware queries.
- Average keyword grounding coverage is 56.9% across `26` keyword-aware queries.
- Hallucination-resistance refusal success is 100.0% across `7` refusal tests.
- Threshold calibration improves supported-answer count at `1.80` compared with baseline `1.20`.