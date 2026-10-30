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

        // m/s と rad/s。1.0 のまま押し続けるとコースには速いので、半分にしてある。
        constexpr double linear_speed = 0.5;
        constexpr double angular_speed = 0.5;

        auto subscribe_callback = [this, linear_speed, angular_speed](const key_event_msgs::msg::KeyEvent::SharedPtr msg) -> void {
            // キーノードは100msごとに、入力が無いとき '\0' を送る。
            // それを停止にすると、押し続けていてもラベルの多くが 0 になる。
            // 入力が無い間は直前の速度を維持して再送する。停止は wasd 以外のキー。
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
                // 前進中の linear.x は残す。w のあとに a を押すと、同時押しなしで弧を描く。
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

// ★【修正】2つのノードを同時に1つのコマンドで立ち上げる処理に書き換えます
int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    
    // 2つのノードを同時に実行するための管理人（Executor）を作ります
    rclcpp::executors::SingleThreadedExecutor exec;
    
    // 1つ目のノード（キー入力を監視してトピックを出すノード）を登録
    auto node1 = std::make_shared<key_event::KeyHitEventNode>();
    exec.add_node(node1);
    
    // 2つ目のノード（あなたの作った、速度に翻訳するノード）を登録
    auto node2 = std::make_shared<TeleopNode>();
    exec.add_node(node2);
    
    // 2つのノードを同時に待機状態（スピン）にします
    exec.spin();
    
    rclcpp::shutdown();
    return 0;
}

