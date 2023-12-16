# run.py
from zenml import pipeline, step


@step
def step_1(path: str) -> str:
    """Step 1 of the pipeline"""
    return "hello" + path

@step
def step_2(input1 : str) -> str:
    """ Sample step 2"""
    return "hello" + input1

@step
def step_3(input1: str) -> str:
    return input1 + "Finished"


@pipeline
def hello_world_pipeline(path: str) -> str:
    response = step_1(path)
    response1 = step_2(response)
    response2 = step_3(response1)
    return response2

if __name__ == "__main__":
    input = 'world'
    output = hello_world_pipeline(input)
    print(output)