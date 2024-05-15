## objects
/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/7


## animal-objects

8
./metrics_syngen/syn/animals_objects/


## CC500

9

## Failure cases

a orange chair and a blue clock
a blue clock and a blue apple
a pink crown and a red chair
a blue crown and a red balloon


a white suitcase and a white chair
a white bow and a white car




## Missing Objects
a blue balloon and a blue apple
a blue crown and a red balloon


## Bad (excite + SynGen)
a white balloon and a white apple






- Syn: Jenshen-Shenon distance
- KL divergence: KL(p_object||q_attribute), mode coverage
- KL divergence: KL(p_attribute||q_object), mode collapse

- prompt2prompt2: KL(p_object||q_object), mode coverage




## Syn (64 images per prompt)

| Distance | CLIP | MIN  | MAX | MEAN| BLIP|
|---|----|----|----|----|----|
|KL(64)|0.355|0.261|0.|0.|0.811|
|KL(4)|0.355|0.263|0.315|0.289|0.814|
|Cosine|0.354|0.261|0.316|0.289|0.812|
|TVN|0.348|0.252|0.318|0.285|0.802|

number = 23, 22

## Cosine(4)
number = 24, 25
| Distance | CLIP | MIN  | MAX | MEAN| BLIP|
|---|----|----|----|----|----|
|Cosine(20.0)|0.354|0.261|0.316|0.289|0.812|
|15.0|0.354|0.260|0.316|0.288|0.812|
|25.0|0.356|0.262|0.316|0.289|0.812|

## Syn+Excite (4 images per prompt)
number = 13, 15, **14**, 16 , 17, 18

| lambdas | CLIP | MIN  | MAX | MEAN| BLIP|
|---|----|----|----|----|----|
|0.1|0.358|0.266|0.315|0.291|0.817|
|0.2| 0.360|0.268|0.315|0.291|0.815|
|0.3| 0.363|0.272|0.313|0.293|0.824|
| **0.5**     | 0.366| 0.273|0.316|0.294| 0.834 |
|1.0      | 0.361| 0.270|0.313|0.291| 0.825|
| 1.5     | 0.360| 0.270|0.314|0.292| 0.818|

Summary: 0.5 is the best

## Syn+SUM (4 images per prompt)
number = 19,20,21

| lambdas | CLIP | MIN  | MAX | MEAN| BLIP|
|---|----|----|----|----|----|
|0.5|0.358|0.266|0.315|0.291|0.817|
|1.0| 0.360|0.268|0.315|0.291|0.815|
|1.5| 0.363|0.272|0.313|0.293|0.824|



# Good results:

A furry spotted bear
A furry spotted rabbit
A skewered strawberry
a spiky bicycle
a teal apple and 
a green guitar
a yellow guitar
a sliced apple
a checkered shoe
a curved spiky car

# Try 
“a green clock in the shape of a pentagon
A yellow book and a red vase.
A black apple and a green backpack
A cube made of brick. A cube with the texture of brick.


a stack of three red cubes with a blue sphere on the right and two green cones on the left.
a cream colored labradoodle next to a white cat with black-tipped ears
A horse sitting on an astronaut’s shoulders.
A rhino beetle this size of a tank grapples a real life passenger airplane on the tarmac


# editing

a yellow guitar -> a green guitar

A yellow book and a red vase -> a red vase and a yellow book

a black apple and a green backpack -> a green backpack and a black apple



photo of a cat riding on a bicycle
photo of a cat riding on a spiky bicycle
photo of a cat riding on a red bicycle
photo of a cat riding on a red spiky bicycle


\begin{table*}[t]
\centering
\caption{Text-Text Similarity and Text-Caption Similarity Comparison of Different Methods on Animal-Animal, Animal-Object, and Object-Object datasets}
\label{tab:comparison}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{Animal-Animal} & \multicolumn{3}{c}{Animal-Object} & \multicolumn{3}{c}{Object-Object} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
 Method  & Full Sim. & Min. Sim. & T-C Sim. & Full  Sim. & Min.  Sim. & T-C Sim. & Full  Sim. & Min. Sim. & T-C Sim. \\
\midrule
SD\cite{Rombach_2022_CVPR} & 0.311 & 0.213 & 0.767 & 0.340 & 0.246 & 0.793 & 0.335 & 0.235 & 0.765 \\
CD\cite{liu2022compositional}& 0.284 & 0.232 & 0.692 & 0.336 & 0.252 & 0.769 & 0.349 & 0.265 & 0.759 \\
StrD\cite{feng2023trainingfree} & 0.306 & 0.210 & 0.761 & 0.336 & 0.242 & 0.781 & 0.332 & 0.234 & 0.762 \\
PtP\cite{hertz2023prompttoprompt} & 0.300 & 0.216 & 0.771 & 0.349 & 0.252 & 0.809 & 0.341 & 0.245 & 0.782 \\
SG\cite{rassin2023linguistic} &- & -&- & 0.355 & 0.264& 0.830 & 0.355 & 0.262 & 0.811\\
SG* & - & - &- & 0.364 & 0.272 & 0.843 \\
AnE\cite{chefer2023attend} & \textbf{0.332} & \textbf{0.248} & \underline{0.806} & 0.353 & 0.265 & 0.830 & 0.360 & 0.270 & 0.811 \\ 
\midrule
Ours & \underline{0.328} & \underline{0.244} & \textbf{0.811} & \textbf{0.355} & \textbf{0.265} &  \textbf{0.835} & 0.357 & 0.269 & 0.806 \\
Ours+AE & - & - & - &\\
\bottomrule
\end{tabular}
\end{table*}

### Ours

## 3 Animals_objects 0.5


## 4 objects

