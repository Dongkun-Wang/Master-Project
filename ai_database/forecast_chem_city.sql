/*
database: enp_ai

*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `forecast_chem_city`
-- ----------------------------
DROP TABLE IF EXISTS `forecast_chem_city`;
CREATE TABLE `forecast_chem_city` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `city_id` int(11) DEFAULT NULL,
  `file_date` datetime DEFAULT NULL,
  `chem_type` varchar(16) DEFAULT NULL,
  `chem_unit` varchar(8) DEFAULT NULL,
  `forecast_time` datetime DEFAULT NULL,
  `chem_value` decimal(10,4) DEFAULT NULL,
  `model` varchar(24) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_fcc_file_date` (`file_date`,`city_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='城市污染物预测表';

