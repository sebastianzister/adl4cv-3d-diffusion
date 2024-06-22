import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np
import pyvista as pv

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

#def visualize_multiple_point_clouds(pcs, labels):
#    fig = go.Figure()
#
#    for i, pc in enumerate(pcs):
#        fig.add_trace(go.Scatter3d(
#            x=pc[:, 0],
#            y=pc[:, 1],
#            z=pc[:, 2],
#            mode='markers',
#            marker=dict(
#                size=2,
#                opacity=0.8
#            ),
#            name=labels[i]
#        ))
#
#    fig.show()

def visualize_multiple_point_clouds(pcs, labels):
    plotter = pv.Plotter()
    plotter.enable_eye_dome_lighting()
    
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    actors = []

    def create_toggle_visibility_callback(actor):
        def toggle_visibility(flag):
            actor.SetVisibility(flag)
        return toggle_visibility

    for i, pc in enumerate(pcs):
        pc_vista = pc.numpy()
        actors.append(plotter.add_points(pc_vista, 
            color=colors[i], 
            #opacity=0.8, 
            render_points_as_spheres=True, 
            point_size=15, 
            name=labels[i]
            ))

        toggle_visibility_callback = create_toggle_visibility_callback(actors[i])
        
        plotter.add_checkbox_button_widget(
            toggle_visibility_callback, 
            color_on=colors[i], 
            value=True,
            position=(10.0, 10.0 + i * 50)
        )
    plotter.show()


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

#def visualize_batch(data, output, target):
#    batch_size = data.shape[0]
#    fig, axs = plt.subplots(batch_size, 3, figsize=(10, 5*batch_size), subplot_kw={'projection': '3d'})
#    
#    for i in range(batch_size):
#        min_val = np.min([data.min(), target.min(), output.min()])
#        max_val = np.max([data.max(), target.max(), output.max()])
#
#        axs[2*i].scatter(target[i, :, 0], target[i, :, 2], target[i, :, 1], c='b', label='Target', s=1)
#        axs[2*i].scatter(data[i, :, 0], data[i, :, 2], data[i, :, 1], c='r', label='Data', s=1)
#        axs[2*i].set_title('Data and Target Point Clouds')
#        axs[2*i].set_xlim([min_val, max_val])
#        axs[2*i].set_ylim([min_val, max_val])
#        axs[2*i].set_zlim([min_val, max_val])
#        
#        axs[2*i + 1].scatter(target[i, :, 0], target[i, :, 2], target[i, :, 1], c='b', label='Target', s=1)
#        axs[2*i + 1].scatter(output[i, :, 0], output[i, :, 2], output[i, :, 1], c='r', label='Output', s=1)
#        axs[2*i + 1].set_title('Output and Target Point Clouds')
#        axs[2*i + 1].set_xlim([min_val, max_val])
#        axs[2*i + 1].set_ylim([min_val, max_val])
#        axs[2*i + 1].set_zlim([min_val, max_val])
#
#        axs[2*i + 2].scatter(output[i, :, 0], output[i, :, 2], output[i, :, 1], c='r', label='Output', s=1)
#        axs[2*i + 2].set_title('Output Point Cloud')
#        axs[2*i + 2].set_xlim([min_val, max_val])
#        axs[2*i + 2].set_ylim([min_val, max_val])
#        axs[2*i + 2].set_zlim([min_val, max_val])
#        
#        axs[2*i].legend()
#        axs[2*i + 1].legend()
#    
#    plt.tight_layout()
#    img = get_img_from_fig(fig)
#    return img

def visualize_batch(data, output, target):
    batch_size = data.shape[0]
    #p = pv.Plotter(shape=(batch_size, 3), window_size=(600*3, 600*batch_size), off_screen=True)
    p = pv.Plotter(shape=(3, batch_size), window_size=(600*batch_size, 600*3), off_screen=True, border=False)
    p.remove_bounding_box()
    
    data = data.numpy()[:,:, [2,0,1]]
    output = output.numpy()[:, :, [2,0,1]]
    target = target.numpy()[:, :, [2,0,1]]
    
    data[:, :, 0] = -data[:, :, 0]
    output[:, :, 0] = -output[:, :, 0]
    target[:, :, 0] = -target[:, :, 0]

    for i in range(batch_size):
        min_val = np.min([data.min(), target.min(), output.min()])
        max_val = np.max([data.max(), target.max(), output.max()])

        p.subplot(0, i)
        p.add_points(target[i], color='lightblue', label='Target', point_size=15, render_points_as_spheres=True)
        p.subplot(1, i)
        p.add_points(output[i], color='lightgreen', label='Output', point_size=15, render_points_as_spheres=True)
        p.subplot(2, i)
        p.add_points(data[i], color='lightcoral', label='Data', point_size=15, render_points_as_spheres=True)

#        p.subplot(i, 0)
#        p.add_points(target[i], color='lightblue', label='Target', point_size=15, render_points_as_spheres=True)
#        p.add_points(data[i], color='lightcoral', label='Data', point_size=15, render_points_as_spheres=True)
#        
#        p.subplot(i, 1)
#        p.add_points(target[i], color='lightblue', label='Target', point_size=15, render_points_as_spheres=True)
#        p.add_points(output[i], color='lightcoral', label='Output', point_size=15, render_points_as_spheres=True)
#
#        p.subplot(i, 2)
#        p.add_points(output[i], color='lightblue', label='Output', point_size=15, render_points_as_spheres=True)
        
    
    
    img = p.screenshot(return_img=True, transparent_background=True, scale=1)
    print(img.dtype, img.shape)
    
    # free memory
    p.close()
    return img.transpose(2, 0, 1)

    