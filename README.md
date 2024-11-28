# AMoE-BTS

> **Adaptive Mixture of Experts for Personalized Modality Importance in Brain Tumor Segmentation(2024)**<br>
> [Jeongeun Kim],[Youngwan Jo], [Sunghyun Ahn], and [Sanghyun Park]<br>

## Overview
<img src="Overiview.png" height="260px"/> 
 Accurate segmentation of brain tumors, especially gliomas, is essential for precise
 diagnosis and treatment planning. Multimodal magnetic resonance imaging scans (T1,
 T1ce, T2, and Flair) provide complementary information about the location, size, and
 characteristics of tumors. Since each modality highlights different aspects of tumor
 information, their effective integration is crucial for precise segmentation. To achieve
 this, three fusion strategies (early, late, and hybrid) have been proposed to combine
 multimodal information. However, existing fusion strategies have two limitations. First,
 the early fusion strategy, which treats multiple modalities as a single input, fails to
 effectively capture intra-modality interactions, whereas late fusion strategy increase
 model complexity and the number of parameters by using separate encoders for each
 modality. Hybrid fusion strategy reduce the number of parameters by sharing weights
 across modalities; however, they do not fully capture the unique modality-specific
 information. Second, previous models have developed complex modality fusion
 modules; however, their complexity typically limits to capture inter-modality
 relationships and efficiency. Toaddress these limitations, we propose an Adaptive
 Mixture of Experts (MoE) for Brain Tumor Segmentation (AMoE-BTS) model, which
 comprises the Local Adaptive MoE (LAM) and Global Adaptive MoE (GAM) blocks.
 The LAM block dynamically extracts modality-specific information. The GAM block then
 captures the complex interactions between different modali ties to improve
 segmentation performance. The experimental results obtained on the BraTS2019 and
 BraTS2020 datasets demonstrate that AMoE-BTS achieves superior performance
 across the WT, TC, and ET regions with high Dice scores and HD95 values. By
 effectively combining the key information provided by each modality, AMoE-BTS
 precisely delineates tumor boundaries, delivering improved segmentation accuracy
 and outperforming state-of-the-art models
