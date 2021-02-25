# Deep Learning Networks from scratch
The idea here is to understand and write each backbone network from scratch and replace them with more state of the art activation functions such as swish or mish.

## Todo
Resnet:

- [x] Resnet50
- [x] Resnet101
- [x] Resnet152

Mobile Net:

- [ ] MobileNetV1
- [ ] MobileNetV2

Utitlities:
- [ ] General purpose function for selecting activation functions
- [ ] Create general purpose module for backbone
- [ ] Pyfile for training, evaluation and demo


## Comparison of Activation Functions
| Activation Function | Matthews Correlation Coefficient |
|---------------------|----------------------------------|
| Relu                | 96.09                            |
| Swish               | 98.04                            |
| Mish                | 98.24                            | 