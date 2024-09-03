# conflict-tasks

This repo is a collection of the code and graphs I thought were most relevant from my investigation into sequential sampling models of conflict-based decision-making tasks. The models I focused on were the **shrinking spotlight model (SSP)** and the **Diffusion Model for Conflict Tasks (DMC, gamma drift)**. They are detailed in this paper (https://www.sciencedirect.com/science/article/abs/pii/S001002851100065X?via%3Dihub) and this paper (https://www.sciencedirect.com/science/article/abs/pii/S0010028515000195?via%3Dihub) respectively. 

# Shrinking spotlight model (SSP) 
SSP is a sequential sampling model of attentional focus in the Eriksen flanker task. It describes attention initially as a large "spotlight" over the flankers that gradually narrows in scope until it is solely focused on the target flanker. This is mathematically operationalized as a normal distribution over the flankers. The **standard deviation (sda)** of the normal distribution gradually decreases by a **linear rate (r)** until the distribution is entirely on the target flanker. The **drift rate (v)** is subsequently the sum of the contribution of attention towards the **outer**,**inner**, and **target** flankers weighted by the **perceptual input (p)** of the flankers respectively. The exact computational description of a simulator of this model is in this repo: https://github.com/AlexanderFengler/ssm-simulators/tree/shrinking-spotlight-model) 

# Diffusion Model for Conflict Tasks (DMC) 

DMC is another sequential sampling model that generalizes beyoned the Eriksen flanker task to conflict-based tasks as a whole. The DMC assumes two possible architectures for conflict-based decision-making: automatic processing of irrelevant stimuli affects controlled processing of relevant stimuli or both automatic and controlled processing converge to affect response times. This model is mathematically operationalized using a **scaled Gamma function**, also described in the github repo stated above as **gamma drift**. 

# LANs Training 
Because both the SSP and the DMC are **mathematically intractable**, I have to use a **likelihood approximation network (LANs)** in order to approximate the likelihood function directly. The repo I used was this one (https://github.com/AlexanderFengler/LAN_pipeline_minimal). The models I trained are saved in **trained LAN data** under **onnx_models**. I trained multiple models for the SSP as the initial bounds of simulated parameter values I had were not reflective of other investigations into the parameter recovery estimability of the SSP (see this paper: https://link.springer.com/article/10.3758/s13423-017-1271-2#Sec10). The **additional onnx model "extra_paramsets"** details my attempt to drastically increase the number of samples simulated and reduce the parameter sets I had initially specified in the other onnx model for SSP. 

# Parameter Recovery 
Parameter recovery estimations for the SSP ranged from poor to fair for most parameters. Parameters **"a"** and **"t"** are reasonably well estimated, but **"r"** and **"ratio"** are particularly poor. Notably, by some mathematical happenstance, the posterior graphs of **"r"** and **"sda"** generally **match** the **correlation** seen in this paper (https://link.springer.com/article/10.3758/s13423-017-1271-2#Sec10) (I also ran the simulations twice on the same parameter boundaries and the trends were consistent across ~200 randomized parameter values). Perhaps this justifies the choice of parameter boundaries, even though I did not notice the same trend in **ratio**. I also noticed that **forward simulations** of **estimated parameters** compared with **true parameters** were quite similar (via Kolmogorov Smirnov tests as well), implying some **model identifiability** issues (you can find these graphs in **data visualization for generated parameter values.ipynb** under **hssm_estimates**). 

Parameter recovery estimations for the DMC were **quite strong** with **"v", "a", "z", and "t"**, but were **quite poor** for parameters specific to the **scaled Gamma function**. 

You can find the rest of the graphs and code I used in the **hssm_testing** folder. You will also find the code I used to generate the parameter values in Brown's supercomputing cluster Oscar in **hssm_posterior_estimates**, along with the generated parameter values themselves. 


