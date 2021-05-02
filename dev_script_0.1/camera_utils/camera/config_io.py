# -*- coding: utf-8 -*-
import configparser

def config_write(setting_dict, settingfile = './setting.ini'):
    config = configparser.ConfigParser()
    
    for section_key in setting_dict:
        section = setting_dict[section_key]['']
        config.add_section(section)

        value_dict = setting_dict[section]
        for value_key in value_dict:
            if value_key == '':
                continue
            config.set(section, str(value_key), str(value_dict[value_key]))
    
    with open(settingfile, 'w') as file:
        config.write(file)


def config_read(settingfile = './setting.ini'):
    # To dict type
    setting_dict = {}
    config = configparser.ConfigParser()
    config.read(settingfile)
    print('Settingfile read  --> '+settingfile)
    for section_key in config.keys():
        setting_dict[section_key] = {}
        print(section_key)
        for value_key in config[section_key]:
            key = config[section_key][value_key]
            setting_dict[section_key][value_key] = key
            print(value_key, ' = ',key)
        print()


if __name__ == "__main__":

    setting_dict = {}
    settingfile = './setting.ini'
    
    setting_dict['camera'] = {}
    setting_dict['camera'][''] = 'camera'
    setting_dict['camera']['auto_focus'] = str(0)
    setting_dict['camera']['port'] = str(111)
    
    setting_dict['production'] = {}
    setting_dict['production'][''] = 'production'
    setting_dict['production']['host'] = 'xxx.co.jp'
    setting_dict['production']['port'] = str(10002)
    config_write(setting_dict)

    config_read(settingfile)
