# chainsaw-robot
Simulating a robotic chainsaw carving a 3D model


## notes 
- It is going to be pretty difficult to first convert a 3D model into a simplified mesh and then determine if the model is cuttable. The easier solution
might be to generate the mesh via cuts that we know are possible, and then re-order the cuts optimally. We can use the projection technique to find possible cuts.

https://arxiv.org/pdf/0912.4540.pdf