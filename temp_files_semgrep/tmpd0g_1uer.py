import React, { Component } from 'react';
import { View, Button } from 'react-native';

interface DefaultProps { 
    onLogout: () => void
}

class SettingsScreen extends Component<DefaultProps, object> {

    handleLogout = () => {
        this.props.onLogout();
    }

    render() {
        return (
            <View>
                <Button 
                    color='tomato'
                    title='Log out'
                    onPress={this.handleLogout}
                />
            </View>
        );
    }
}

export default SettingsScreen;