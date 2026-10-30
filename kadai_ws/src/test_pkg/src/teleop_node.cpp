#include <functional>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "key_event_msgs/msg/key_event.hpp"
#include "key_event_nodes/key_hit_event_node.hpp"

using namespace std::chrono_literals;

class TeleopNode : public rclcpp::Node {
public:
    //送信用のメッセージを宣言
    geometry_msgs::msg::Twist message = geometry_msgs::msg::Twist();

    TeleopNode(
        const std::string& name_space="", 
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    ) : Node("teleop_node",name_space,options) {
        
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        constexpr double linear_speed = 0.5;
        constexpr double angular_speed = 0.5;

        auto subscribe_callback = [this, linear_speed, angular_speed](const key_event_msgs::msg::KeyEvent::SharedPtr msg) -> void {
            if (msg->key == '\0') {
                this->publisher_->publish(message);
                return;
            }

            //キーボードの入力から進路を決定
            switch (msg->key)
            {
            //cahr型なのでシングルクオーテーションを用いる必要があります
            case 's':
                message.linear.x = -linear_speed;
                message.angular.z = 0;
                break;
            case 'w':
                message.linear.x = linear_speed;
                message.angular.z = 0;
                break;
            case 'a':
                message.angular.z = angular_speed;
                break;
            case 'd':
                message.angular.z = -angular_speed;
                break;
            default:
                message.linear.x = 0;
                message.angular.z = 0;
                break;
            }

            this->publisher_->publish(message);
        }; 

        //キーボードの値取得用のsubscriber
        sub_ = this->create_subscription<key_event_msgs::msg::KeyEvent>(
        "key_hit_event",
        rclcpp::QoS(10),
        subscribe_callback
        );
    }
private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::Subscription<key_event_msgs::msg::KeyEvent>::SharedPtr sub_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);

    rclcpp::executors::SingleThreadedExecutor exec;

    auto node1 = std::make_shared<key_event::KeyHitEventNode>();
    exec.add_node(node1);

    auto node2 = std::make_shared<TeleopNode>();
    exec.add_node(node2);

    exec.spin();
    
    rclcpp::shutdown();
    return 0;
}

