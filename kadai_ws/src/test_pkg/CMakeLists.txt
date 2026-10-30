cmake_minimum_required(VERSION 3.8)
project(test_pkg)


# 1. 依存パッケージを検索（★ key_event_nodes を追加）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(key_event_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(key_event_nodes REQUIRED)

# 2. 実行ファイルの作成
add_executable(teleop_node src/teleop_node.cpp)

# 3. 依存関係を確実に紐付ける（★ key_event_nodes を追加）
ament_target_dependencies(teleop_node
  rclcpp
  key_event_msgs
  std_msgs
  geometry_msgs
  rclcpp_components
  key_event_nodes
)

# 4. インストール設定（ビルド後に実行できるようにする）
install(TARGETS
  teleop_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

#//install(DIRECTORY launch models
#//  DESTINATION share/${PROJECT_NAME}
#//)
install(DIRECTORY models launch worlds
  DESTINATION share/${PROJECT_NAME}
)
ament_package()

