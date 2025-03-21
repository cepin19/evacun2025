# evacun2025

Code for training, running and evaluating CUNI submission to the EvaCun 2025 token restoration shared task.
Our best checkpoint can be downloaded from: https://huggingface.co/cepin/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit_evacun_task

After saving the model into directory model/, you can run evaluation on our held-out dev set:

`cat dev.csv | python -u predict.py --model unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit  --peft_model model/ --mode all  --output_json predictions.json &> predictions.out `

To aggregate multiple prediction files and compute the metrics, run:
`python aggregate.py predictions*.json`

The model was trained by the folowing command:

`python llm_finetune.py --base_model=unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit  --save_dir=model/"  --r 16`
