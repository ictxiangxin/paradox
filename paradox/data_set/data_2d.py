import numpy


def helical_data(number: int,
                 category: int=2,
                 noise: float=0,
                 max_radius: int=5,
                 max_theta: float=3*numpy.pi,
                 init_radius: float=0,
                 init_theta: float=0):
    radius_step = max_radius / number
    theta_step = max_theta / number
    phase = 2 * numpy.pi / category
    data = [[[], []] for _ in range(category)]
    start_phase = [i * phase for i in range(category)]
    radius = init_radius
    theta = init_theta
    for _ in range(number):
        for i in range(category):
            data[i][0].append(radius * numpy.cos(theta + start_phase[i]) + numpy.random.normal(0, noise))
            data[i][1].append(radius * numpy.sin(theta + start_phase[i]) + numpy.random.normal(0, noise))
        radius += radius_step
        theta += theta_step
    return data
