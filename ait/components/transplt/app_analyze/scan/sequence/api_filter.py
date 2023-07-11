from app_analyze.common.kit_config import KitConfig

GLOBAL_FILTER = ['operator=', 'operator>>', 'operator<<', 'operator[]', 'operator()',
                 'operator!=', 'operator*', 'operator->', 'operator*=', 'operator++',
                 'operator&=', 'operator+=', 'operator>=', 'operator,', 'operator+',
                 'operator<', 'operator>', 'operator/', 'operator==', 'operator-',
                 'operator&', 'operator!']
_opencv_filter = ['cv::Mat.ptr', 'cv::Mat.at']
ACC_FILTER = {KitConfig.OPENCV: _opencv_filter}
