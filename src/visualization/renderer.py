"""
Rendering Module for Track-MDP Visualization

This module provides pygame-based visualization for the tracking environment,
showing the grid, sensors, and object position in real-time.

Based on the original Track-MDP.py rendering implementation.
"""

import numpy as np
import pygame
import os


class Sprite(pygame.sprite.Sprite):
    """
    Sprite class for rendering grid elements, sensors, and robot.
    Matches the original Track-MDP.py implementation.
    """
    def __init__(self, x, y, size, color, image='Square', add_image=None):
        super().__init__()
        if image == 'Square':
            self.image = pygame.Surface([size, size])
            self.image.fill(color)
        else:
            self.image = pygame.image.load(image)
        
        if add_image is not None:
            self.original_image = pygame.image.load(image)
            self.add_image = pygame.image.load(add_image)
        
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
    def move(self, center):
        self.rect.x = center[0]
        self.rect.y = center[1]
    
    def turn_on(self):
        self.image = self.add_image
    
    def turn_off(self):
        self.image = self.original_image
    
    def update(self):
        self.image = self.original_image


class TrackingRenderer:
    """
    A renderer for visualizing the Track-MDP environment using pygame.
    Matches the original Track-MDP.py rendering implementation exactly.
    
    Attributes:
        N (int): Grid size (NxN)
        sq_pixels (int): Size of each grid cell in pixels
        screen_size (int): Total screen size
        fps (int): Frames per second
        screen: Pygame screen surface
        sprites: Pygame sprite groups for rendering
    """
    
    def __init__(self, grid_size=10, sq_pixels=40, fps=5, title='Track-MDP'):
        """
        Initialize the renderer.
        
        Args:
            grid_size (int): Size of the grid (default: 10)
            sq_pixels (int): Pixels per grid cell (default: 40)
            fps (int): Frames per second (default: 5)
            title (str): Window title (default: 'Track-MDP')
        """
        self.N = grid_size
        self.sq_pixels = sq_pixels
        self.screen_size = self.N * self.sq_pixels
        self.fps = fps
        self.title = title
        
        # Initialize pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        
        # Create display
        height = self.screen_size
        width = self.screen_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        
        # Grid parameters (matching original)
        W = self.N
        H = self.N
        self.sq_size = min(height // H, width // W)
        
        # Sprite groups
        self.rects = pygame.sprite.Group()
        self.sensors = pygame.sprite.Group()
        self.sensor_list = []
        
        # Offsets
        offsetx = width - W * self.sq_size
        offsety = height - H * self.sq_size
        self.offx = offsetx
        self.offy = offsety
        
        # Build grid (matching original exactly)
        thickness = 0.95
        
        # Get asset paths
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets')
        robot_image = os.path.join(assets_dir, 'robot4.png')
        sensor_off_image = os.path.join(assets_dir, 'antenna_off.png')
        sensor_on_image = os.path.join(assets_dir, 'antenna_on.png')
        
        # Load sensor image to get dimensions
        sensor_image = pygame.image.load(sensor_off_image)
        sensor_rect = sensor_image.get_rect()
        
        # Create robot sprite
        self.robot = Sprite(
            x=self.sq_size + offsetx // 2,
            y=self.sq_size + offsety // 2,
            size=self.sq_size,
            color=(0, 0, 0),
            image=robot_image
        )
        self.main_sprites = pygame.sprite.Group(self.robot)
        
        # Create grid cells and sensors (matching original)
        for i in range(W):
            for j in range(H):
                # Background square
                temp_square = Sprite(
                    x=i * self.sq_size + offsetx // 2,
                    y=j * self.sq_size + offsety // 2,
                    size=self.sq_size,
                    color=(0, 0, 0)
                )
                self.rects.add(temp_square)
                
                # Inner cell (gray)
                t = 125
                self.rects.add(Sprite(
                    x=i * self.sq_size + offsetx // 2 + (1 - thickness) / 2 * self.sq_size,
                    y=j * self.sq_size + offsety // 2 + (1 - thickness) / 2 * self.sq_size,
                    size=thickness * self.sq_size,
                    color=(t, t, t)
                ))
                
                # Sensor sprite
                temp_sensor = Sprite(
                    x=i * self.sq_size + offsetx // 2 + (1 - thickness) / 2 * self.sq_size + sensor_rect.width // 2,
                    y=j * self.sq_size + offsety // 2 + (1 - thickness) / 2 * self.sq_size + sensor_rect.height // 2,
                    size=thickness * self.sq_size,
                    color=(t, t, t),
                    image=sensor_off_image,
                    add_image=sensor_on_image
                )
                self.sensor_list.append(temp_sensor)
                self.sensors.add(self.sensor_list[-1])
        
        self.graphical = True
        self.anim_action = None
    
    def update_sensors(self, anim_action, object_pos):
        """
        Update sensor display based on action.
        Matches original Track-MDP.py implementation.
        
        Args:
            anim_action (np.ndarray): NxN array of sensor activations
            object_pos (int): Flattened position of object
        """
        self.anim_action = anim_action
        self.object_pos = object_pos
    
    def render(self, wait_for_input=False):
        """
        Render the current frame.
        Matches original Track-MDP.py rendering exactly.
        
        Args:
            wait_for_input (bool): If True, wait for user input before continuing
            
        Returns:
            bool: False if user closed window, True otherwise
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if wait_for_input and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        if self.anim_action is None:
            return True
        
        # Update sensors (matching original)
        self.sensors.update()
        for sensor in range(self.N ** 2):
            j, i = np.unravel_index(sensor, shape=(self.N, self.N))
            if self.anim_action[i, j] == 1:
                self.sensor_list[sensor].turn_on()
            else:
                self.sensor_list[sensor].turn_off()
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid and sensors
        self.rects.draw(self.screen)
        self.sensors.draw(self.screen)
        
        # Draw robot/object at current position
        if self.object_pos < self.N ** 2:
            pos = np.unravel_index(self.object_pos, shape=(self.N, self.N))
            coords = int(pos[0]), int(pos[1])
            x = coords[1] * self.sq_size
            y = coords[0] * self.sq_size
            center = x + self.robot.rect.width // 1.5, y + self.robot.rect.height // 1.5
            self.robot.move(center)
            self.main_sprites.draw(self.screen)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(self.fps)
        
        # Wait for input if requested
        if wait_for_input:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False
                        waiting = False
        
        return True
    
    def close(self):
        """Close the renderer and clean up pygame."""
        pygame.quit()


# Legacy sprite classes for backwards compatibility
class GridCell(pygame.sprite.Sprite):
    """Legacy GridCell sprite (kept for compatibility)."""
    def __init__(self, x, y, size, inner_size, border):
        super().__init__()
        self.image = pygame.Surface((size, size))
        self.image.fill((0, 0, 0))
        pygame.draw.rect(self.image, (125, 125, 125), (border, border, inner_size, inner_size))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class SensorSprite(pygame.sprite.Sprite):
    """Legacy SensorSprite (kept for compatibility)."""
    def __init__(self, x, y, size, active=True):
        super().__init__()
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        if active:
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            overlay.fill((0, 255, 0, 100))
            self.image.blit(overlay, (0, 0))
            center = size // 2
            pygame.draw.circle(self.image, (0, 255, 0), (center, center), size // 6)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class ObjectSprite(pygame.sprite.Sprite):
    """Legacy ObjectSprite (kept for compatibility)."""
    def __init__(self, x, y, size, color=(255, 0, 0)):
        super().__init__()
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        radius = size // 3
        pygame.draw.circle(self.image, color, (center, center), radius)
        pygame.draw.circle(self.image, (255, 255, 255), (center, center), radius, 2)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y