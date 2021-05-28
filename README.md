# Paper: Argument Undermining: Counter-Argument generation by Attacking Weak Premises

This is the code for the paper *Argument Undermining: Counter-Argument generation by Attacking Weak Premises*.

Milad Alshomary, Shahbaz Syed, Martin Potthast, and Henning Wachsmuth


      @InProceedings{alshomary:2021b,
        author =              {Milad Alshomary, Shahbaz Syed, Martin Potthast, and Henning Wachsmuth},
        booktitle =           {The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)},
        month =               August,
        publisher =           {ACL-IJCNLP},
        title =               {{Argument Undermining: Counter-Argument generation by Attacking Weak Premises}},
        year =                2021
      }

-----------------------------------------------

## Data:
The data folder consists of the following:
  - _predictions_: contains all counter-generation predictions of our approach and the baselines.
  - _evaluations_: contains annotations results for all our manual evaluation studies.
  - _cmv_dataset_: would be made available soon.
## Code:
  - _prepare_ds.ipynb_ : Contains the code for processing Jo et al. 2020 dataset and collecting counter-arguments. For this notebook to work, you need the data provided in Jo et al. 2020.
The code folder contains all python file and notebooks necessary to reproduce our results:
  - _premise_attackability_: Contains the code for training our LTR-bert model (ltr_identify_vunerability.ipynb) and the commands needed to re-train BERT-classifier of Jo et al. 2020. For these notebooks to work, you need the tensorflow ranking library (https://github.com/tensorflow/ranking) and the code of Jo et al (https://github.com/yohanjo/emnlp20_arg_attack)
  
  - _attack_generation_: Contains the code for fine-tuning GPT model and the baseline, as well as generting attacks on argument given the weak-premises.
  
  - _overall_approach.ipynb_: Contains our code for the overall approach, that is identifying weak premises using ltr-bert and then generate attacks.
  
  - _evaluation_: Contains all notebooks for evaluating all steps of our approach
  
  
