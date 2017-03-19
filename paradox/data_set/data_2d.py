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


def grid_data(number: int,
              noise: float=0,
              raw: int=2,
              column: int=2,
              width: float=5,
              hight: float=5,
              ):
    data = [[[], []], [[], []]]
    raw_step = hight / raw
    column_step = width / column
    for _ in range(number):
        for i in range(raw):
            for j in range(column):
                x = numpy.random.random() * column_step
                y = numpy.random.random() * raw_step
                data[(i + j) % 2][0].append(x + column_step * j + numpy.random.normal(0, noise))
                data[(i + j) % 2][1].append(y + raw_step * i + numpy.random.normal(0, noise))
    return data
