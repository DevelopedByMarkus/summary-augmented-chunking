from tqdm.auto import tqdm
import datasets

from tasks import TASKS, ISSUE_TASKS
from utils import generate_prompts


def main():

    # Supress progress bars which appear every time a task is downloaded
    # datasets.utils.logging.set_verbosity_error()

    # Debug
    print(len(TASKS), TASKS[:10])
    print()
    print(len(ISSUE_TASKS), ISSUE_TASKS)

    # download the datasets
    dataset = datasets.load_dataset("nguha/legalbench", "abercrombie")
    dataset["train"].to_pandas()
    print(dataset)

    # Load base prompt
    with open(f"tasks/abercrombie/base_prompt.txt") as in_file:
        prompt_template = in_file.read()
    print(prompt_template)

    # Create full prompts
    test_df = dataset["test"].to_pandas()
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)
    print(prompts[0])

    # Evaluation
    from evaluation import evaluate
    import numpy as np

    # Generate random predictions for abercrombie
    classes = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]
    generations = np.random.choice(classes, len(test_df))

    evaluate("abercrombie", generations, test_df["answer"].tolist())

    # Select tasks by licence
    target_license = "CC BY 4.0"
    tasks_with_target_license = []
    for task in tqdm(TASKS):
        dataset = datasets.load_dataset("nguha/legalbench", task, split="train")
        if dataset.info.license == target_license:
            tasks_with_target_license.append(task)
    print("Tasks with target license:", tasks_with_target_license)


if __name__ == "__main__":
    main()
