from util import load_all_data, set_system_parameters, create_model


data = load_all_data()

desired_behaviour = False
while not desired_behaviour:

    params = set_system_parameters()

    model = create_model(params)

    model.fit(data, params)

    # validation = model.validate()
    # desired_behaviour = some_condition(validation)
