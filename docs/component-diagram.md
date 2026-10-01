# Component Diagram

This diagram maps the internal relationships and dependencies between the core Python modules in the system.

```mermaid
classDiagram
    class App_py {
        +Flask app
        +routes()
    }
    class Src_Occupancy {
        +calculate_overlap()
        +classify_slots()
    }
    class Src_Parking_Slots {
        +load_config()
        +validate_polygons()
    }
    class Src_Visualization {
        +draw_slots()
        +annotate_frame()
    }
    class Src_Vehicle_Detector {
        +load_model()
        +detect_vehicles()
    }
    class Src_Video_Processor {
        +process_video_stream()
        +extract_frames()
    }
    class Src_Analytics {
        +generate_stats()
        +format_chart_data()
    }

    App_py ..> Src_Occupancy : uses
    App_py ..> Src_Visualization : uses
    App_py ..> Src_Vehicle_Detector : uses
    App_py ..> Src_Video_Processor : uses
    App_py ..> Src_Analytics : uses
    
    Src_Occupancy ..> Src_Parking_Slots : depends on
    Src_Visualization ..> Src_Occupancy : depends on
    Src_Visualization ..> Src_Parking_Slots : depends on
    
    Src_Video_Processor ..> Src_Vehicle_Detector : coordinates
    Src_Video_Processor ..> Src_Occupancy : coordinates
    Src_Video_Processor ..> Src_Visualization : coordinates
    Src_Video_Processor ..> Src_Analytics : coordinates
```

## Component Roles
- **app.py**: The main entry point. Coordinates web requests and bridges frontend to backend logic.
- **src.occupancy**: Handles spatial mathematics (intersection logic) to determine space status.
- **src.parking_slots**: Manages the loading, parsing, and validation of user-configured JSON polygon definitions.
- **src.visualization**: Centralized module for all OpenCV drawing commands (lines, text, colors).
- **src.vehicle_detector**: Wrapper around the Ultralytics YOLOv8 API.
- **src.video_processor**: Orchestrates the frame-by-frame analysis loop for video files.
- **src.analytics**: Compiles detection data into structured JSON formats optimized for frontend Chart.js rendering.
