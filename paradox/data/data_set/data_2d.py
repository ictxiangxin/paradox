import numpy


def helical_data(number: int,
                 category: int=2,
                 noise: float=0,
                 max_radius: int=5,
                 max_theta: float=3*numpy.pi,
                 init_radius: float=0,
                 init_theta: float=0,
                 center: list=[0, 0]):
    radius_step = max_radius / number
    theta_step = max_theta / number
    phase = 2 * numpy.pi / category
    data = [[[], []] for _ in range(category)]
    start_phase = [i * phase for i in range(category)]
    radius = init_radius
    theta = init_theta
    x0, x1 = center
    for _ in range(number):
        for i in range(category):
            data[i][0].append(x0 + radius * numpy.cos(theta + start_phase[i]) + numpy.random.normal(0, noise))
            data[i][1].append(x1 + radius * numpy.sin(theta + start_phase[i]) + numpy.random.normal(0, noise))
        radius += radius_step
        theta += theta_step
    return data


def grid_data(number: int,
              noise: float=0,
              raw: int=2,
              column: int=2,
              width: float=5,
              height: float=5,
              ):
    data = [[[], []], [[], []]]
    raw_step = height / raw
    column_step = width / column
    for _ in range(number):
        for i in range(raw):
            for j in range(column):
                x = numpy.random.random() * column_step
                y = numpy.random.random() * raw_step
                data[(i + j) % 2][0].append(x + column_step * j + numpy.random.normal(0, noise))
                data[(i + j) % 2][1].append(y + raw_step * i + numpy.random.normal(0, noise))
    return data


def circle_data(number: int,
                category: int=2,
                center: list=[0, 0],
                delta_radius: float=1.,
                noise: float=0.
                ):
    data = [[[], []] for _ in range(category)]
    radius = 0.
    theta_step = 2. * numpy.pi / number
    x0, x1 = center
    for i in range(category):
        radius += delta_radius
        theta = 0.
        for _ in range(number):
            data[i][0].append(x0 + radius * numpy.cos(theta) + numpy.random.normal(0, noise))
            data[i][1].append(x1 + radius * numpy.sin(theta) + numpy.random.normal(0, noise))
            theta += theta_step
    return data


def gaussian_data(number: int,
                  category: int=2,
                  mean: list=None,
                  cov: list=None):
    data = []
    mean = mean or [[i + 1, i + 1] for i in range(category)]
    cov = cov or [numpy.eye(2) for _ in range(category)]
    for i in range(category):
        data.append(numpy.random.multivariate_normal(mean[i], cov[i], number).T)
    return data


def cross_data(number: int,
               category: int=2,
               center: list=[0, 0],
               radius: float=2,
               init_angle: float=0.,
               noise: float=0.):
    delta_angle = numpy.pi / category
    data = [[[], []] for _ in range(category)]
    x0, x1 = center
    for i in range(category):
        x_step = 2. * radius * numpy.cos(init_angle) / number
        y_step = 2. * radius * numpy.sin(init_angle) / number
        x_start = x0 + radius * numpy.cos(init_angle)
        y_start = x1 + radius * numpy.sin(init_angle)
        for _ in range(number):
            data[i][0].append(x_start + numpy.random.normal(0, noise))
            data[i][1].append(y_start + numpy.random.normal(0, noise))
            x_start -= x_step
            y_start -= y_step
        init_angle += delta_angle
    return data