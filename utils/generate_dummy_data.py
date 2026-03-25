import numpy as np

samples = []

for i in range(50):

    events = np.zeros((500,4))

    events[:,0] = np.random.randint(0,128,500)
    events[:,1] = np.random.randint(0,128,500)
    events[:,2] = np.linspace(0,1,500)
    events[:,3] = np.random.choice([-1,1],500)

    label = np.random.randint(0,5)

    samples.append({
        "events": events,
        "label": label
    })


np.save("data/train.npy", samples)

print("Dummy dataset created")