# README: Training Configs for Pistachio Splits

This includes the config files used for OOD Pistachio splits. If you have access to the Pistachio dataset, please
edit the paths as appropriate after using our reaction splitting code.

Note that the author-document based split only contains the base config used for 
hyperparameter tuning (as the checkpoint for the optimal run can be taken directly from the hyperparameter run directory
and no other models are required to be trained using these parameters for this evaluation).

For the NameRxn and time-based splits we only include the main configs. 
The other configs can be derived from the ones provided here by overriding the data paths. (These configs can be 
derived programmatically if required).