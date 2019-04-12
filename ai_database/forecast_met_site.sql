/*
database: enp_ai
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `forecast_met_site`
-- ----------------------------
DROP TABLE IF EXISTS `forecast_met_site`;
CREATE TABLE `forecast_met_site` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `site_id` int(11) DEFAULT NULL,
  `file_date` datetime DEFAULT NULL,
  `met_type` varchar(16) DEFAULT NULL,
  `met_unit` varchar(8) DEFAULT NULL,
  `forecast_time` datetime DEFAULT NULL,
  `met_value` decimal(10,4) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='站点气象预测表';

SET FOREIGN_KEY_CHECKS = 1;
