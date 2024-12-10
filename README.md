# VTMo: Unified Visuo-Tactile Transformer Encoder with Mixture-of-Modality-Experts

This repository contains the code and datasets for the paper **"VTMo: Unified Visuo-Tactile Transformer Encoder with Mixture-of-Modality-Experts"**, which introduces a modular Vision-Touch Transformer encoder designed to unify the strengths of dual-encoder and fusion-encoder architectures for visuo-tactile modeling tasks.

## Overview
Paper: https://www.zichenz.me/project/vtmo/vtmo.pdf

### Key Contributions:
- **Modular Architecture**: Inspired by VLMo used in language and vision, VTMo integrates modality-specific and cross-modal experts within a shared attention mechanism.
- **Versatility**: Functions as a single-modality encoder, a dual-encoder, or a fusion encoder, depending on the task.
- **Efficiency and Accuracy**: Achieves competitive accuracy with reduced computation and faster convergence compared to baseline models.

### Applications:
- **Image-to-Touch Retrieval**: Demonstrated competitive performance and efficiency on the Touch and Go dataset.
- **Potential Extensions**: Future applications include X-to-Touch generation and image synthesis using touch.
