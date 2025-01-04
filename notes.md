# Explanation of files and stuff I looked at:
## First tasks until intermediate report 

1) plane graphs, preliminary tests with inefficient algorithm. .npz saving 
    -> gen_graph.py

    -> test.ipynb

    -> graphs/

2) plane graphs, final code with dataset creation 
    -> gen_dataset.py

    -> test_dataset.ipynb

    -> plane_graphs/

3) 3D graphs 
    -> gen_3D_dataset.py

    -> 3d_tests.ipynb

    -> 3D_graphs/

## After intermediate report: defect detection 

4) first defect detection on 3D graphs
    -> gen_defect_dataset.py

    -> defect_tests.ipynb

    -> defect_graphs/
    
    NOTE: This still used the idea of nopde prediction. The code is not working or up to date, as this idea has quickly been discontinued. The class for class creation is for instance defined twice, once in the test file, once in the original creation file. Bo9th are not the same. The NN does not work.

5) Defect detection with autoencoders on 3D-graphs

