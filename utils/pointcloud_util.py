import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np

def visualize_point_cloud(pc):
	fig = go.Figure(data=[go.Scatter3d(
	    x=pc[:, 0],
	    y=pc[:, 1],
	    z=pc[:, 2],
	    mode='markers',
	    marker=dict(
	        size=2,
	        #color=samples[i][:, 2],  # color points by z-axis value
	        #colorscale='Viridis',
	        opacity=0.8
	    )
	)])
	
	fig.update_layout(scene=dict(
	    xaxis_title='X',
	    yaxis_title='Y',
	    zaxis_title='Z'
	))
	
	fig.show()

def visualize_multiple_point_clouds(pcs, labels):
    fig = go.Figure()

    for i, pc in enumerate(pcs):
        fig.add_trace(go.Scatter3d(
            x=pc[:, 0],
            y=pc[:, 1],
            z=pc[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.8
            ),
            name=labels[i]
        ))

    fig.show()

def compare_point_cloud(pc1, pc2):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=pc1[:, 0],
        y=pc1[:, 1],
        z=pc1[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        ),
        name='Point Cloud 1'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=pc2[:, 0],
        y=pc2[:, 1],
        z=pc2[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.8
        ),
        name='Point Cloud 2'
    ))
    
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    
    fig.show()

def get_img_from_fig(fig, dpi=180):
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1][0], fig.canvas.get_width_height()[::-1][1], 3)
    return data.transpose((2, 0, 1))

def visualize_batch(data, output, target):
    batch_size = data.shape[0]
    fig, axs = plt.subplots(batch_size, 3, figsize=(10, 5*batch_size), subplot_kw={'projection': '3d'})
    
    for i in range(batch_size):
        min_val = np.min([data.min(), target.min(), output.min()])
        max_val = np.max([data.max(), target.max(), output.max()])

        axs[2*i].scatter(target[i, :, 0], target[i, :, 2], target[i, :, 1], c='b', label='Target', s=1)
        axs[2*i].scatter(data[i, :, 0], data[i, :, 2], data[i, :, 1], c='r', label='Data', s=1)
        axs[2*i].set_title('Data and Target Point Clouds')
        axs[2*i].set_xlim([min_val, max_val])
        axs[2*i].set_ylim([min_val, max_val])
        axs[2*i].set_zlim([min_val, max_val])
        
        axs[2*i + 1].scatter(target[i, :, 0], target[i, :, 2], target[i, :, 1], c='b', label='Target', s=1)
        axs[2*i + 1].scatter(output[i, :, 0], output[i, :, 2], output[i, :, 1], c='r', label='Output', s=1)
        axs[2*i + 1].set_title('Output and Target Point Clouds')
        axs[2*i + 1].set_xlim([min_val, max_val])
        axs[2*i + 1].set_ylim([min_val, max_val])
        axs[2*i + 1].set_zlim([min_val, max_val])

        axs[2*i + 2].scatter(output[i, :, 0], output[i, :, 2], output[i, :, 1], c='r', label='Output', s=1)
        axs[2*i + 2].set_title('Output Point Cloud')
        axs[2*i + 2].set_xlim([min_val, max_val])
        axs[2*i + 2].set_ylim([min_val, max_val])
        axs[2*i + 2].set_zlim([min_val, max_val])
        
        axs[2*i].legend()
        axs[2*i + 1].legend()
    
    plt.tight_layout()
    img = get_img_from_fig(fig)
    return img

    