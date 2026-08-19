import torch


from src.model.model_pipeline import ModelPipeline



def create_model():

    model = ModelPipeline(
        input_dim=51,
        state_dim=6
    )

    return model



def test_pipeline_output():


    model = create_model()


    model.eval()


    x = torch.randn(
        2,
        20,
        51
    )


    with torch.no_grad():

        output = model(
            x
        )


    assert output is not None



    if isinstance(output, dict):

        assert "state" in output


        state = output["state"]


    else:

        state = output



    assert state.shape[0] == 2

    assert state.shape[-1] == 6




def test_pipeline_no_nan():


    model = create_model()


    x = torch.randn(
        2,
        20,
        51
    )


    output = model(
        x
    )


    if isinstance(output, dict):

        state = output["state"]

    else:

        state = output



    assert not torch.isnan(
        state
    ).any()



def test_pipeline_gradient():


    model = create_model()


    x = torch.randn(
        2,
        20,
        51,
        requires_grad=True
    )


    output = model(
        x
    )


    if isinstance(output, dict):

        state = output["state"]

    else:

        state = output



    loss = state.mean()


    loss.backward()



    has_grad=False


    for p in model.parameters():

        if p.grad is not None:

            has_grad=True

            break



    assert has_grad
