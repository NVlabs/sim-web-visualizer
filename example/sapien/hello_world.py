import sapien.core as sapien
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from sim_web_visualizer import create_sapien_visualizer, bind_visualizer_to_sapien_scene


def main():
    # create simulation engine
    engine = sapien.Engine()

    # Create renderer
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    # Create a simulation scene
    create_sapien_visualizer(port=6000, host="localhost", keep_default_viewer=False)
    scene = engine.create_scene()
    scene = bind_visualizer_to_sapien_scene(scene, engine, renderer)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # Add actors(rigid bodies)
    scene.add_ground(altitude=0)  # Add a ground
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1.0, 0.0, 0.0])
    box = actor_builder.build(name="box")  # Add a box
    box.set_pose(sapien.Pose(p=[0, 0, 0.5]))

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    #############################
    while True:
        try:
            scene.step()  # Simulate the world
            scene.update_render()  # Update the world to the renderer

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    start_zmq_server_as_subprocess()
    main()
