import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

nbElements = 5

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
plt.xlim(0, 1)
plt.ylim(-0.08, 0.12)
plt.xlabel('Elements')
plt.ylabel('Fitness')
plt.title('Fitness Distribution')

ax1.set_xlim(0, 1.05)
ax1.set_ylim(-0.08, 0.2)

# Rectangle principale
ax1.add_patch(Rectangle((0, 0), 1, 0.1, linewidth=3, edgecolor='black', facecolor='white'))

tab_Rec = []
linewidth = 0.002
element_width = 1 / nbElements  # Width of each element slot

# Draw black and red rectangles
for i in range(nbElements):
    black_width = element_width * 0.3  # Black rectangle takes 30% of the element slot width
    red_width = element_width * 0.7    # Red rectangle takes 70% of the element slot width

    black_rect = Rectangle((i * element_width, 0), black_width, 0.1, linewidth=linewidth, edgecolor='black', facecolor='black')
    red_rect = Rectangle((i * element_width + black_width, 0), red_width, 0.1, linewidth=linewidth, edgecolor='black', facecolor='green')

    ax1.add_patch(black_rect)
    ax1.add_patch(red_rect)
    ax1.text((i + 0.5) / nbElements, -0.02, str(i + 1), fontsize=8, ha='center')

    tab_Rec.append(red_rect)

plt.show()
